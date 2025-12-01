#ifndef UTEC_NN_LOSS_H
#define UTEC_NN_LOSS_H

#include "nn_interfaces.h"
#include <cmath>
#include "nn_loss.h"
using utec::algebra::Tensor;
namespace utec::neural_network {


template <typename T, size_t dim = 2>
class MSELoss final : public ILoss<T, dim> {
    static_assert(dim == 2, "Este proyecto solo usa tensores 2-D");
private:
    Tensor<T, dim> pred, target;

public:
    MSELoss(const Tensor<T, dim>& y_pred, const Tensor<T, dim>& y_true)
        : pred(y_pred), target(y_true) {}

    T loss() const override {
        T sum = 0;
        const size_t total = pred.shape()[0] * pred.shape()[1];
        for (size_t i = 0; i < pred.shape()[0]; ++i)
            for (size_t j = 0; j < pred.shape()[1]; ++j) {
                const T d = pred(i,j) - target(i,j);
                sum += d * d;
            }
        return sum / static_cast<T>(total);
    }

    Tensor<T, dim> loss_gradient() const override {
        Tensor<T, dim> grad(pred.shape());
        const size_t total = pred.shape()[0] * pred.shape()[1];
        for (size_t i = 0; i < grad.shape()[0]; ++i)
            for (size_t j = 0; j < grad.shape()[1]; ++j)
                grad(i,j) =
                    static_cast<T>(2) * (pred(i,j) - target(i,j))
                    / static_cast<T>(total);
        return grad;
    }
};

template <typename T, size_t dim = 2>
class BCELoss final : public ILoss<T, dim> {
    static_assert(dim == 2, "Este proyecto solo usa tensores 2-D");
private:
    Tensor<T, dim> pred, target;

public:
    BCELoss(const Tensor<T, dim>& y_pred, const Tensor<T, dim>& y_true)
        : pred(y_pred), target(y_true) {}

    T loss() const override {
        const T eps = 1e-12;
        T sum = 0;
        const size_t total = pred.shape()[0] * pred.shape()[1];
        for (size_t i = 0; i < pred.shape()[0]; ++i)
            for (size_t j = 0; j < pred.shape()[1]; ++j) {
                T y = target(i,j);
                T p = std::clamp(pred(i,j), eps, 1 - eps);
                sum += -y * std::log(p) - (1 - y) * std::log(1 - p);
            }
        return sum / static_cast<T>(total);
    }

    Tensor<T, dim> loss_gradient() const override {
        const T eps = 1e-12;
        Tensor<T, dim> grad(pred.shape());
        const size_t total = pred.shape()[0] * pred.shape()[1];
        for (size_t i = 0; i < grad.shape()[0]; ++i)
            for (size_t j = 0; j < grad.shape()[1]; ++j) {
                T y = target(i,j);
                T p = std::clamp(pred(i,j), eps, 1 - eps);
                grad(i,j) =
                    (p - y) / (p * (1 - p) * static_cast<T>(total));
            }
        return grad;
    }
};

} // namespace utec::neural_network
#endif // UTEC_NN_LOSS_H