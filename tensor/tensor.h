#ifndef UTEC_ALGEBRA_TENSOR_H
#define UTEC_ALGEBRA_TENSOR_H

#include <array>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <initializer_list>
#include <type_traits>

using namespace std;

namespace utec {
namespace algebra {

template <typename T, size_t Rank>
class Tensor {
    static_assert(Rank >= 1, "El rango debe ser al menos 1");

public:
    Tensor() = default;

    explicit Tensor(const array<size_t, Rank>& shape) {
        _shape = shape;
        _total_size = compute_total_size(_shape);
        _data.resize(_total_size);
    }

    template <typename... Dims>
    explicit Tensor(Dims... dims) {
        vector<size_t> dimsVec = { static_cast<size_t>(dims)... };
        if (dimsVec.size() != Rank) {
            throw invalid_argument("Numero de dimensiones no concuerda con " + to_string(Rank));
        }
        array<size_t, Rank> shp;
        for (size_t i = 0; i < Rank; ++i) {
            shp[i] = dimsVec[i];
        }
        _shape = shp;
        _total_size = compute_total_size(_shape);
        _data.resize(_total_size);
    }

    Tensor& operator=(initializer_list<T> list) {
        if (_total_size == 0) {
            throw invalid_argument("Tensor shape must be set before assigning values");
        }
        if (list.size() != _total_size) {
            throw invalid_argument("Tamano de datos no concuerda con el tamano del tensor");
        }
        copy(list.begin(), list.end(), _data.begin());
        return *this;
    }

    array<size_t, Rank> shape() const noexcept {
        return _shape;
    }
    size_t size() const noexcept { return _total_size; }
    auto begin() noexcept { return _data.begin(); }
    auto end() noexcept { return _data.end(); }
    auto cbegin() const noexcept { return _data.cbegin(); }
    auto cend() const noexcept { return _data.cend(); }

    void fill(const T& value) noexcept {
        fill_n(_data.begin(), _total_size, value);
    }

    void reshape(const array<size_t, Rank>& new_shape) {
        size_t new_total = compute_total_size(new_shape);
        if (new_total == _total_size) {
            _shape = new_shape;
            return;
        }
        if (new_total < _total_size) {
            vector<T> temp(new_total);
            for (size_t i = 0; i < new_total; ++i) {
                temp[i] = _data[i];
            }
            _data = move(temp);
            _total_size = new_total;
            _shape = new_shape;
            return;
        }
        _data.resize(new_total);
        _total_size = new_total;
        _shape = new_shape;
    }

    template <typename... Dims>
    void reshape(Dims... dims) {
        vector<size_t> dimsVec = { static_cast<size_t>(dims)... };
        if (dimsVec.size() != Rank) {
            throw invalid_argument("Numero de dimensiones no conucerda con " + to_string(Rank));
        }
        array<size_t, Rank> new_shape;
        for (size_t i = 0; i < Rank; ++i) {
            new_shape[i] = dimsVec[i];
        }
        size_t new_total = compute_total_size(new_shape);
        if (new_total == _total_size) {
            _shape = new_shape;
            return;
        }
        if (new_total < _total_size) {
            vector<T> temp(new_total);
            for (size_t i = 0; i < new_total; ++i) {
                temp[i] = _data[i];
            }
            _data = move(temp);
            _total_size = new_total;
            _shape = new_shape;
            return;
        }
        _data.resize(new_total);
        _total_size = new_total;
        _shape = new_shape;
    }

    template <typename... Idxs>
    T& operator()(Idxs... idxs) {
        vector<size_t> idxVec = { static_cast<size_t>(idxs)... };
        if (idxVec.size() != Rank) {
            throw invalid_argument("Numero de indices no conucerda con " + to_string(Rank));
        }
        array<size_t, Rank> indices;
        for (size_t i = 0; i < Rank; ++i) {
            indices[i] = idxVec[i];
        }
        size_t offset = compute_offset(indices);
        return _data[offset];
    }

    template <typename... Idxs>
    const T& operator()(Idxs... idxs) const {
        vector<size_t> idxVec = { static_cast<size_t>(idxs)... };
        if (idxVec.size() != Rank) {
            throw invalid_argument("Numero de indices no conucerda " + to_string(Rank));
        }
        array<size_t, Rank> indices;
        for (size_t i = 0; i < Rank; ++i) {
            indices[i] = idxVec[i];
        }
        size_t offset = compute_offset(indices);
        return _data[offset];
    }

    T& operator()(const array<size_t, Rank>& indices) {
        size_t offset = compute_offset(indices);
        return _data[offset];
    }

    const T& operator()(const array<size_t, Rank>& indices) const {
        size_t offset = compute_offset(indices);
        return _data[offset];
    }

    T& operator[](size_t idx) {
        if (Rank != 2) {
            throw invalid_argument("operator[] solo disponible para tensor 2d");
        }
        if (idx >= _total_size) {
            throw invalid_argument("Indice fuera de rango);
        }
        size_t cols = _shape[1];
        size_t i = idx / cols;
        size_t j = idx % cols;
        return operator()(i, j);
    }

    const T& operator[](size_t idx) const {
        if (Rank != 2) {
            throw invalid_argument("operator[] solo disponible para tensor 2d");
        }
        if (idx >= _total_size) {
            throw invalid_argument("Indice fuera de rango");
        }
        size_t cols = _shape[1];
        size_t i = idx / cols;
        size_t j = idx % cols;
        return operator()(i, j);
    }

    Tensor operator+(const Tensor& other) const {
        return elementwise_binary_op(other, plus<>{});
    }
    Tensor operator-(const Tensor& other) const {
        return elementwise_binary_op(other, minus<>{});
    }
    Tensor operator*(const Tensor& other) const {
        return elementwise_binary_op(other, multiplies<>{});
    }

    Tensor operator+(const T& scalar) const {
        Tensor result(_shape);
        for (size_t i = 0; i < _total_size; ++i) {
            result._data[i] = _data[i] + scalar;
        }
        return result;
    }
    Tensor operator-(const T& scalar) const {
        Tensor result(_shape);
        for (size_t i = 0; i < _total_size; ++i) {
            result._data[i] = _data[i] - scalar;
        }
        return result;
    }
    Tensor operator*(const T& scalar) const {
        Tensor result(_shape);
        for (size_t i = 0; i < _total_size; ++i) {
            result._data[i] = _data[i] * scalar;
        }
        return result;
    }
    Tensor operator/(const T& scalar) const {
        Tensor result(_shape);
        for (size_t i = 0; i < _total_size; ++i) {
            result._data[i] = _data[i] / scalar;
        }
        return result;
    }

    friend Tensor operator+(const T& scalar, const Tensor& t) {
        return t + scalar;
    }
    friend Tensor operator-(const T& scalar, const Tensor& t) {
        Tensor result(t._shape);
        for (size_t i = 0; i < t._total_size; ++i) {
            result._data[i] = scalar - t._data[i];
        }
        return result;
    }
    friend Tensor operator*(const T& scalar, const Tensor& t) {
        return t * scalar;
    }
    friend Tensor operator/(const T& scalar, const Tensor& t) {
        Tensor result(t._shape);
        for (size_t i = 0; i < t._total_size; ++i) {
            result._data[i] = scalar / t._data[i];
        }
        return result;
    }

    friend ostream& operator<<(ostream& os, const Tensor& t) {
        if constexpr (Rank == 1) {
            for (size_t i = 0; i < t._shape[0]; ++i) {
                os << t._data[i];
                if (i + 1 < t._shape[0]) os << " ";
            }
            os << "\n";
            return os;
        } else {
            array<size_t, Rank> idx{};
            t.print_recursive(os, idx, 0);
            return os;
        }
    }

    static array<size_t, Rank> linear_to_multi(size_t linear, const array<size_t, Rank>& shape) {
        array<size_t, Rank> indices{};
        size_t remainder = linear;
        size_t stride = compute_total_size(shape);
        for (size_t dim = 0; dim < Rank; ++dim) {
            stride /= shape[dim];
            indices[dim] = remainder / stride;
            remainder %= stride;
        }
        return indices;
    }

private:
    array<size_t, Rank> _shape{};
    vector<T> _data;
    size_t _total_size{0};

    static size_t compute_total_size(const array<size_t, Rank>& shape) {
        return accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                          multiplies<size_t>{});
    }

    size_t compute_offset(const array<size_t, Rank>& indices) const {
        size_t offset = 0;
        size_t stride = 1;
        for (size_t dim = Rank; dim-- > 0;) {
            if (indices[dim] >= _shape[dim]) {
                throw invalid_argument("Indice fuera del rango de dimension " + to_string(dim));
            }
            offset += indices[dim] * stride;
            stride *= _shape[dim];
        }
        return offset;
    }

    template <typename Op>
    Tensor elementwise_binary_op(const Tensor& other, Op op) const {
        array<size_t, Rank> result_shape{};
        for (size_t i = 0; i < Rank; ++i) {
            size_t a_dim = _shape[i];
            size_t b_dim = other._shape[i];
            if (a_dim == b_dim) {
                result_shape[i] = a_dim;
            } else if (a_dim == 1) {
                result_shape[i] = b_dim;
            } else if (b_dim == 1) {
                result_shape[i] = a_dim;
            } else {
                throw invalid_argument(
                    "Shapes do not match and they are not compatible for broadcasting");
            }
        }
        Tensor result(result_shape);
        size_t result_total = result._total_size;
        for (size_t idx = 0; idx < result_total; ++idx) {
            auto idx_multi = linear_to_multi(idx, result_shape);
            array<size_t, Rank> idxA{}, idxB{};
            for (size_t i = 0; i < Rank; ++i) {
                idxA[i] = (_shape[i] == 1 ? 0 : idx_multi[i]);
                idxB[i] = (other._shape[i] == 1 ? 0 : idx_multi[i]);
            }
            T a_val = this->operator()(idxA);
            T b_val = other.operator()(idxB);
            result._data[idx] = op(a_val, b_val);
        }
        return result;
    }

    void print_recursive(ostream& os, array<size_t, Rank>& idx, size_t dim) const {
        os << "{\n";
        if (dim == Rank - 2) {
            size_t rows = _shape[dim];
            size_t cols = _shape[dim + 1];
            for (size_t i = 0; i < rows; ++i) {
                idx[dim] = i;
                for (size_t j = 0; j < cols; ++j) {
                    idx[dim + 1] = j;
                    os << operator()(idx) << " ";
                }
                os << "\n";
            }
        } else {
            size_t limit = _shape[dim];
            for (size_t i = 0; i < limit; ++i) {
                idx[dim] = i;
                print_recursive(os, idx, dim + 1);
            }
        }
        os << "}\n";
    }
};

template <typename T, size_t R>
Tensor<T, R> transpose_2d(const Tensor<T, R>& t) {
    if constexpr (R < 2) {
        throw invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
    }
    auto old_shape = t.shape();
    array<size_t, R> new_shape = old_shape;
    new_shape[R - 2] = old_shape[R - 1];
    new_shape[R - 1] = old_shape[R - 2];
    Tensor<T, R> result(new_shape);
    size_t total = accumulate(new_shape.begin(), new_shape.end(), static_cast<size_t>(1),
                              multiplies<size_t>{});
    for (size_t idx = 0; idx < total; ++idx) {
        auto idx_multi = Tensor<T, R>::linear_to_multi(idx, new_shape);
        array<size_t, R> idx_orig = idx_multi;
        swap(idx_orig[R - 2], idx_orig[R - 1]);
        result(idx_multi) = t(idx_orig);
    }
    return result;
}

template <typename T, size_t R>
Tensor<T, R> matrix_product(const Tensor<T, R>& A, const Tensor<T, R>& B) {
    if constexpr (R < 2) {
        throw invalid_argument("Dimensiones de la matriz no son compatibles para multiplicar");
    }
    auto a_shape = A.shape();
    auto b_shape = B.shape();
    for (size_t i = 0; i < R - 2; ++i) {
        if (a_shape[i] != b_shape[i]) {
            throw invalid_argument(
                "Dimensiones de la matriz son compatibles para multiplicar pero las dimensiones de lote no coinciden");

        }
    }
    size_t M = a_shape[R - 2];
    size_t K = a_shape[R - 1];
    size_t K2 = b_shape[R - 2];
    size_t N = b_shape[R - 1];
    if (K != K2) {
        throw invalid_argument("Dimensiones de la matriz no son compatibles para multiplicar");
    }
    array<size_t, R> result_shape = a_shape;
    result_shape[R - 2] = M;
    result_shape[R - 1] = N;
    Tensor<T, R> result(result_shape);
    size_t total = accumulate(result_shape.begin(), result_shape.end(), static_cast<size_t>(1),
                              multiplies<size_t>{});
    for (size_t idx = 0; idx < total; ++idx) {
        auto idx_multi = Tensor<T, R>::linear_to_multi(idx, result_shape);
        array<size_t, R> idxA = idx_multi;
        array<size_t, R> idxB = idx_multi;
        T sum = T{};
        for (size_t k = 0; k < K; ++k) {
            idxA[R - 2] = idx_multi[R - 2];
            idxA[R - 1] = k;
            idxB[R - 2] = k;
            idxB[R - 1] = idx_multi[R - 1];
            sum += A(idxA) * B(idxB);
        }
        result(idx_multi) = sum;
    }
    return result;
}

} // namespace algebra
} // namespace utec

#endif // UTEC_ALGEBRA_TENSOR_H