#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <memory>
#include "tensor.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_optimizer.h"
#include "nn_interfaces.h"

using namespace std;

using Float = double;
using Tensor2 = utec::algebra::Tensor<Float, 2>;

//semilla random
mt19937_64 rng(time(nullptr));

void guardar_csv(const vector<Example>& data, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("No se pudo crear CSV: " + filename);
    }

    for (const auto& e : data) {
        for (size_t i = 0; i < e.x.size(); i++) {
            file << e.x[i] << ",";
        }
        file << e.label << "\n";
    }
}

double gauss(double s = 1.0) {
    static normal_distribution<double> dist(0.0, 1.0);
    return dist(rng) * s;
}

double uniform(double a, double b) {
    uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

//data structures
struct Example {
    vector<Float> x;
    int label;
};

vector<Example> generar_formas(int N, int cant_por_clase) {
    vector<Example> data;
    const double PI = 3.14159265358979;
    const double TWO_PI = 2 * PI;

    for (int clase = 0; clase < 3; ++clase) {
        for (int s = 0; s < cant_por_clase; ++s) {
            vector<Float> v(N);
            double rot = uniform(0, TWO_PI);
            double escala = uniform(0.9, 1.1);

            for (int i = 0; i < N; i++) {
                double ang = (TWO_PI * i / N) + rot;
                double r = 0;

                if (clase == 0) { // circulo
                    r = 1.0;
                }
                else if (clase == 1) { // cuadrado
                    double c = fabs(cos(ang));
                    double s2 = fabs(sin(ang));
                    double h = 1.0 / sqrt(2.0);
                    double d = max(c, s2);
                    r = h / d;
                }
                else if (clase == 2) { // triangulo
                    double t = fmod(ang, (2.0 * PI / 3.0));
                    r = fabs(cos(PI / 3.0 - t));
                }

                r = r * escala + gauss(0.03);
                if (r < 0) r = 0;
                v[i] = r;
            }

            // normalizacion
            double mx = *max_element(v.begin(), v.end());
            if (mx <= 0) mx = 1.0;
            for (auto &a : v) a /= mx;

            data.push_back({v, clase});
        }
    }

    shuffle(data.begin(), data.end(), rng);
    return data;
}

Example generar_figura_unica(int N, int clase) {
    const double PI = 3.14159265358979;
    const double TWO_PI = 2 * PI;

    vector<Float> v(N);
    double rot = uniform(0, TWO_PI);
    double escala = uniform(0.9, 1.1);

    for (int i = 0; i < N; i++) {
        double ang = (TWO_PI * i / N) + rot;
        double r = 0;

        if (clase == 0) { // circulo
            r = 1.0;
        }
        else if (clase == 1) { // cuadrado
            double c = fabs(cos(ang));
            double s2 = fabs(sin(ang));
            double h = 1.0 / sqrt(2.0);
            double d = max(c, s2);
            r = h / d;
        }
        else { // triangulo
            double t = fmod(ang, (2.0 * PI / 3.0));
            r = fabs(cos(PI / 3.0 - t));
        }

        r = r * escala + gauss(0.03);
        if (r < 0) r = 0;
        v[i] = r;
    }

    double mx = *max_element(v.begin(), v.end());
    if (mx <= 0) mx = 1.0;
    for (auto &a : v) a /= mx;

    return {v, clase};
}

Tensor2 batch_X(const vector<Example>& batch) {
    size_t B = batch.size();
    size_t N = batch[0].x.size();

    Tensor2 X({B, N});
    for (size_t i = 0; i < B; i++)
        for (size_t j = 0; j < N; j++)
            X(i,j) = batch[i].x[j];
    return X;
}

Tensor2 batch_Y(const vector<Example>& batch, int C) {
    size_t B = batch.size();
    Tensor2 Y({B, (size_t)C});

    for (size_t i = 0; i < B; i++) {
        for (int c = 0; c < C; c++) Y(i,c) = 0;
        Y(i, batch[i].label) = 1.0;
    }

    return Y;
}

Tensor2 softmax(const Tensor2 &Z) {
    auto shape = Z.shape();
    size_t B = shape[0], C = shape[1];
    Tensor2 P({B, C});

    for (size_t i = 0; i < B; i++) {
        double m = Z(i,0);
        for (size_t c = 1; c < C; c++)
            m = max(m, Z(i,c));

        double suma = 0;
        for (size_t c = 0; c < C; c++) {
            P(i,c) = exp(Z(i,c) - m);
            suma += P(i,c);
        }
        for (size_t c = 0; c < C; c++)
            P(i,c) /= suma;
    }
    return P;
}

double cross_entropy(const Tensor2& P, const Tensor2& Y) {
    size_t B = P.shape()[0], C = P.shape()[1];
    double loss = 0;

    for (size_t i = 0; i < B; i++)
        for (size_t c = 0; c < C; c++)
            if (Y(i,c) > 0.5)
                loss += -log(max(1e-12, P(i,c)));

    return loss / B;
}

Tensor2 grad_softmax_CE(const Tensor2 &P, const Tensor2 &Y) {
    auto shape = P.shape();
    Tensor2 G(shape);

    size_t B = shape[0], C = shape[1];
    for (size_t i = 0; i < B; i++)
        for (size_t c = 0; c < C; c++)
            G(i,c) = (P(i,c) - Y(i,c)) / B;

    return G;
}

using namespace utec::neural_network;

struct MLP {
    unique_ptr<ILayer<Float>> L1, A1, L2, A2, L3;

    MLP(size_t n) {
        auto initW = [&](Tensor2 &W){
            normal_distribution<double> d(0,0.1);
            for (size_t i = 0; i < W.size(); i++) W[i] = d(rng);
        };
        auto initB = [&](Tensor2 &b){
            for (size_t i = 0; i < b.size(); i++) b[i] = 0.0;
        };

        L1 = make_unique<Dense<Float>>(n, 64, initW, initB);
        A1 = make_unique<ReLU<Float>>();
        L2 = make_unique<Dense<Float>>(64, 32, initW, initB);
        A2 = make_unique<ReLU<Float>>();
        L3 = make_unique<Dense<Float>>(32, 3, initW, initB);
    }

    Tensor2 forward(const Tensor2 &X) {
        return L3->forward(
                A2->forward(
                    L2->forward(
                        A1->forward(
                            L1->forward(X)
                        )
                    )
                )
        );
    }

    void backward(const Tensor2 &g) {
        auto g2 = L3->backward(g);
        auto g3 = A2->backward(g2);
        auto g4 = L2->backward(g3);
        A1->backward(g4);
    }

    void update(utec::neural_network::Adam<Float> &opt) {
        L1->update_params(opt);
        L2->update_params(opt);
        L3->update_params(opt);
    }
};

int main() {
    int N = 32;
    auto data = generar_formas(N, 2000);

    guardar_csv(data, "dataset_generado.csv");
    cout << "Dataset generado y guardado en dataset_generado.csv\n";

    size_t trainN = static_cast<size_t>(data.size() * 0.8);
    vector<Example> train(data.begin(), data.begin() + trainN);
    vector<Example> val(data.begin() + trainN, data.end());

    MLP model(N);
    utec::neural_network::Adam<Float> opt(0.001);

    int epochs = 60;
    int batch = 32;

    for (int ep = 0; ep < epochs; ep++) {
        shuffle(train.begin(), train.end(), rng);

        for (size_t i = 0; i < train.size(); i += batch) {
            size_t fin = min(train.size(), i + batch);
            vector<Example> B(train.begin() + i, train.begin() + fin);

            Tensor2 X = batch_X(B);
            Tensor2 Y = batch_Y(B, 3);

            Tensor2 Z = model.forward(X);
            Tensor2 P = softmax(Z);
            Tensor2 G = grad_softmax_CE(P, Y);

            model.backward(G);
            model.update(opt);
        }

        size_t correct = 0;
        double loss = 0;

        for (size_t i = 0; i < val.size(); i += batch) {
            size_t fin = min(val.size(), i + batch);
            vector<Example> B(val.begin() + i, val.begin() + fin);

            Tensor2 X = batch_X(B);
            Tensor2 Y = batch_Y(B, 3);

            Tensor2 P = softmax(model.forward(X));
            loss += cross_entropy(P, Y) * (fin - i);

            for (size_t r = 0; r < P.shape()[0]; r++) {
                int pred = 0;
                double m = P(r,0);
                for (int c = 1; c < 3; c++)
                    if (P(r,c) > m) { m = P(r,c); pred = c; }

                if (pred == B[r].label) correct++;
            }
        }

        loss /= val.size();
        double acc = double(correct) / val.size();

        cout << "Epoca " << ep
             << " | loss=" << loss
             << " | acc=" << acc << endl;
    }

    cout << "\nEntrenamiento terminado.\n";

    cout << "\nPrueba de prediccion: \n";

    int figura = rand() % 3;

    cout << "Figura generada: ";
    if (figura == 0) cout << "Circulo\n";
    else if (figura == 1) cout << "Cuadrado\n";
    else cout << "Triangulo\n";

    Example e = generar_figura_unica(N, figura);
    vector<Example> prueba = { e };

    Tensor2 Xtest = batch_X(prueba);
    Tensor2 logits = model.forward(Xtest);
    Tensor2 P = softmax(logits);

    int pred = 0;
    double maxp = P(0,0);
    for (int c = 1; c < 3; c++) {
        if (P(0,c) > maxp) {
            maxp = P(0,c);
            pred = c;
        }
    }

    cout << "Prediccion del modelo: ";
    if (pred == 0) cout << "Circulo";
    else if (pred == 1) cout << "Cuadrado";
    else cout << "Triangulo";

    cout << "\nProbabilidades: ["
         << P(0,0) << ", " << P(0,1) << ", " << P(0,2) << "]";

    return 0;
}
