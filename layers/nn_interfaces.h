#ifndef UTEC_NN_INTERFACES_H
#define UTEC_NN_INTERFACES_H

#include "tensor.h"
using utec::algebra::Tensor;

namespace utec::neural_network {

    template <typename T>
    class IOptimizer {
    public:
        virtual ~IOptimizer() = default;

        virtual void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) = 0;
        virtual void step() {}
    };

    template <typename T>
    class ILayer {
    public:
        virtual ~ILayer() = default;

        virtual Tensor<T, 2> forward(const Tensor<T, 2>& x) = 0;
        virtual Tensor<T, 2> backward(const Tensor<T, 2>& grad) = 0;

        virtual void update_params(IOptimizer<T>& optimizer) {}
    };

    template <typename T, size_t dim>
    class ILoss {
    public:
        virtual ~ILoss() = default;

        virtual T loss() const = 0;
        virtual Tensor<T, dim> loss_gradient() const = 0;
    };

} // namespace utec::neural_network

#endif // UTEC_NN_INTERFACES_H