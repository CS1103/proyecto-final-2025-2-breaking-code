#ifndef UTEC_NN_OPTIMIZER_H
#define UTEC_NN_OPTIMIZER_H

#include "nn_interfaces.h"
#include <unordered_map>
#include <cmath>

using utec::algebra::Tensor;

namespace utec::neural_network {

template <typename T>
class SGD final : public IOptimizer<T> {
private:
    T lr;

public:
    explicit SGD(T learning_rate = 0.01) : lr(learning_rate) {}

    void step() override {}                          // no lleva contador

    void update(Tensor<T,2>& params,
                const Tensor<T,2>& grads) override {
        for (size_t i = 0; i < params.shape()[0]; ++i)
            for (size_t j = 0; j < params.shape()[1]; ++j)
                params(i,j) -= lr * grads(i,j);
    }
};


template <typename T>
class Adam final : public IOptimizer<T> {
private:
    T lr, beta1, beta2, epsilon;
    size_t t;                                                //contador de pasos
    std::unordered_map<Tensor<T,2>*, Tensor<T,2>> m_map;     //momento 1
    std::unordered_map<Tensor<T,2>*, Tensor<T,2>> v_map;     //momento 2

public:
    explicit Adam(T learning_rate = 0.001,
                  T beta1 = 0.9,
                  T beta2 = 0.999,
                  T eps   = 1e-8)
        : lr(learning_rate), beta1(beta1),
          beta2(beta2), epsilon(eps), t(0) {}

    void step() override {}

    void update(Tensor<T,2>& param,
                const Tensor<T,2>& grad) override {

        ++t;

        auto& m = m_map[&param];
        auto& v = v_map[&param];

        if (m.shape() != param.shape())
            m = Tensor<T,2>(param.shape());
        if (v.shape() != param.shape())
            v = Tensor<T,2>(param.shape());

        for (size_t i = 0; i < param.shape()[0]; ++i) {
            for (size_t j = 0; j < param.shape()[1]; ++j) {

                m(i,j) = beta1 * m(i,j) + (1 - beta1) * grad(i,j);
                v(i,j) = beta2 * v(i,j) + (1 - beta2) * grad(i,j) * grad(i,j);

                const T m_hat = m(i,j) /
                                (1 - std::pow(beta1, static_cast<T>(t)));
                const T v_hat = v(i,j) /
                                (1 - std::pow(beta2, static_cast<T>(t)));

                param(i,j) -= lr * m_hat /
                              (std::sqrt(v_hat) + epsilon);
            }
        }
    }
};

} // namespace utec::neural_network

#endif // UTEC_NN_OPTIMIZER_H