#ifndef UTEC_NN_ACTIVATION_H
#define UTEC_NN_ACTIVATION_H

#include "nn_interfaces.h"
#include <cmath>
using utec::algebra::Tensor;
namespace utec::neural_network {

    template <typename T>
    class ReLU final : public ILayer<T> {
    private:
        Tensor<T, 2> mask;

    public:
        Tensor<T, 2> forward(const Tensor<T, 2>& z) override {
            mask = Tensor<T, 2>(z.shape());
            Tensor<T, 2> result(z.shape());
            for (size_t i = 0; i < z.shape()[0]; ++i) {
                for (size_t j = 0; j < z.shape()[1]; ++j) {
                    if (z(i,j) > 0) {
                        result(i,j) = z(i,j);
                        mask(i,j) = 1;
                    } else {
                        result(i,j) = 0;
                        mask(i,j) = 0;
                    }
                }
            }
            return result;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& g) override {
            Tensor<T, 2> grad(g.shape());
            for (size_t i = 0; i < g.shape()[0]; ++i) {
                for (size_t j = 0; j < g.shape()[1]; ++j) {
                    grad(i,j) = g(i,j) * mask(i,j);
                }
            }
            return grad;
        }
    };

    template <typename T>
    class Sigmoid final : public ILayer<T> {
    private:
        Tensor<T, 2> output;

    public:
        Tensor<T, 2> forward(const Tensor<T, 2>& z) override {
            output = Tensor<T, 2>(z.shape());
            for (size_t i = 0; i < z.shape()[0]; ++i) {
                for (size_t j = 0; j < z.shape()[1]; ++j) {
                    output(i,j) = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-z(i,j)));
                }
            }
            return output;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& g) override {
            Tensor<T, 2> grad(g.shape());
            for (size_t i = 0; i < g.shape()[0]; ++i) {
                for (size_t j = 0; j < g.shape()[1]; ++j) {
                    grad(i,j) = g(i,j) * output(i,j) * (1 - output(i,j));
                }
            }
            return grad;
        }
    };

} // namespace utec::neural_network

#endif // UTEC_NN_ACTIVATION_H