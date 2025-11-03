#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <thread>
#include <future>
#include <numeric>
#include <limits>

static size_t MULTITHREADING_THRESHOLD = 500;  // Use multithreading (even if enabled) only if range is greater or equal to this

class Matrix {
private:
    std::vector<double> data_;
    size_t rows_;
    size_t cols_;
    size_t num_threads_;
    bool enable_checks_;
    bool use_multithreading_;

    // Private helper methods
    void validate_dimensions(const Matrix& other) const;
    void validate_multiplication(const Matrix& other) const;
    void check_nan_inf() const;

    template<typename Func>
    void parallel_for(size_t start, size_t end, Func&& func) const;

public:
    // Constructors
    Matrix(size_t rows, size_t cols, double init_val = 0.0, size_t num_threads = 1, bool enable_checks = false);
    Matrix(size_t rows, size_t cols, const std::vector<double>& values, size_t num_threads = 1, bool enable_checks = false);

    // Accessors
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    size_t rows() const;
    size_t cols() const;
    size_t size() const;
    void set_num_threads(size_t n);
    void set_enable_checks(bool enable);
    void set_use_multithreading(bool enable);
    void set_multithreading_threshold(size_t n);

    // Element-wise operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix element_wise_multiply(const Matrix& other) const;
    Matrix element_wise_divide(const Matrix& other) const;

    // Scalar operations
    Matrix operator+(double scalar) const;
    Matrix operator-(double scalar) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    // In-place operations
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);

    // Matrix operations
    Matrix transpose() const;
    double trace() const;
    Matrix row(size_t idx) const;
    Matrix col(size_t idx) const;

    // Statistical operations
    double min() const;
    double max() const;
    double sum() const;
    double mean() const;
    double variance() const;
    double std_dev() const;

    // Element-wise math functions
    Matrix sqrt() const;
    Matrix exp() const;
    Matrix log() const;
    Matrix pow(double exponent) const;
    Matrix abs() const;
    Matrix sin() const;
    Matrix cos() const;
    Matrix tan() const;

    // Utility functions
    void fill(double value);
    Matrix reshape(size_t new_rows, size_t new_cols) const;
    const std::vector<double>& get_data() const;
    std::vector<double>& get_data();
};

// ==================== IMPLEMENTATIONS ====================

// Private helper methods
inline void Matrix::validate_dimensions(const Matrix& other) const {
    if (enable_checks_ && (rows_ != other.rows_ || cols_ != other.cols_)) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
}

inline void Matrix::validate_multiplication(const Matrix& other) const {
    if (enable_checks_ && cols_ != other.rows_) {
        throw std::invalid_argument("Invalid dimensions for multiplication");
    }
}

inline void Matrix::check_nan_inf() const {
    if (!enable_checks_) return;
    for (const auto& val : data_) {
        if (std::isnan(val) || std::isinf(val)) {
            throw std::runtime_error("Matrix contains NaN or Inf values");
        }
    }
}

template<typename Func>
inline void Matrix::parallel_for(size_t start, size_t end, Func&& func) const {
    // If multithreading is disabled or range is too small, execute serially
    if (!use_multithreading_ || num_threads_ <= 1 || (end - start) <= MULTITHREADING_THRESHOLD) {
        func(start, end);
        return;
    }

    size_t chunk_size = (end - start + num_threads_ - 1) / num_threads_;
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads_; ++i) {
        size_t chunk_start = start + i * chunk_size;
        size_t chunk_end = std::min(chunk_start + chunk_size, end);
        if (chunk_start >= end) break;

        futures.push_back(std::async(std::launch::async, func, chunk_start, chunk_end));
    }

    for (auto& f : futures) {
        f.get();
    }
}

// Constructors
inline Matrix::Matrix(size_t rows, size_t cols, double init_val, size_t num_threads, bool enable_checks)
    : data_(rows* cols, init_val), rows_(rows), cols_(cols),
    num_threads_(num_threads), enable_checks_(enable_checks),
    use_multithreading_(num_threads > 1) {
    
}

inline Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& values, size_t num_threads, bool enable_checks)
    : data_(values), rows_(rows), cols_(cols),
    num_threads_(num_threads), enable_checks_(enable_checks),
    use_multithreading_(num_threads > 1) {
    if (enable_checks_ && data_.size() != rows_ * cols_) {
        throw std::invalid_argument("Vector size must match matrix dimensions");
    }
}

// Accessors
inline double& Matrix::operator()(size_t row, size_t col) {
    return data_[row * cols_ + col];
}

inline const double& Matrix::operator()(size_t row, size_t col) const {
    return data_[row * cols_ + col];
}

inline size_t Matrix::rows() const { return rows_; }
inline size_t Matrix::cols() const { return cols_; }
inline size_t Matrix::size() const { return data_.size(); }
inline void Matrix::set_num_threads(size_t n) {
    num_threads_ = n;
    use_multithreading_ = (n > 1);
}
inline void Matrix::set_enable_checks(bool enable) { enable_checks_ = enable; }
inline void Matrix::set_use_multithreading(bool enable) { use_multithreading_ = enable; }
inline void Matrix::set_multithreading_threshold(size_t n) { MULTITHREADING_THRESHOLD = n; }

// Element-wise operations
inline Matrix Matrix::operator+(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        });

    return result;
}

inline Matrix Matrix::operator-(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        });

    return result;
}

inline Matrix Matrix::operator*(const Matrix& other) const {
    validate_multiplication(other);
    Matrix result(rows_, other.cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, rows_, [&](size_t start_row, size_t end_row) {
        for (size_t i = start_row; i < end_row; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        });

    return result;
}

inline Matrix Matrix::element_wise_multiply(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        });

    return result;
}

inline Matrix Matrix::element_wise_divide(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            if (enable_checks_ && other.data_[i] == 0.0) {
                throw std::runtime_error("Division by zero");
            }
            result.data_[i] = data_[i] / other.data_[i];
        }
        });

    return result;
}

// Scalar operations
inline Matrix Matrix::operator+(double scalar) const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] + scalar;
        }
        });

    return result;
}

inline Matrix Matrix::operator-(double scalar) const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] - scalar;
        }
        });

    return result;
}

inline Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        });

    return result;
}

inline Matrix Matrix::operator/(double scalar) const {
    if (enable_checks_ && scalar == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] / scalar;
        }
        });

    return result;
}

// In-place operations
inline Matrix& Matrix::operator+=(const Matrix& other) {
    validate_dimensions(other);
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data_[i] += other.data_[i];
        }
        });
    return *this;
}

inline Matrix& Matrix::operator-=(const Matrix& other) {
    validate_dimensions(other);
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data_[i] -= other.data_[i];
        }
        });
    return *this;
}

inline Matrix& Matrix::operator*=(double scalar) {
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data_[i] *= scalar;
        }
        });
    return *this;
}

inline Matrix& Matrix::operator/=(double scalar) {
    if (enable_checks_ && scalar == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data_[i] /= scalar;
        }
        });
    return *this;
}

// Matrix operations
inline Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, rows_, [&](size_t start_row, size_t end_row) {
        for (size_t i = start_row; i < end_row; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        });

    return result;
}

inline double Matrix::trace() const {
    if (enable_checks_ && rows_ != cols_) {
        throw std::invalid_argument("Trace requires square matrix");
    }

    double sum = 0.0;
    for (size_t i = 0; i < rows_; ++i) {
        sum += (*this)(i, i);
    }
    return sum;
}

inline Matrix Matrix::row(size_t idx) const {
    if (enable_checks_ && idx >= rows_) {
        throw std::out_of_range("Row index out of range");
    }
    std::vector<double> row_data(data_.begin() + idx * cols_, data_.begin() + (idx + 1) * cols_);
    return Matrix(1, cols_, row_data, num_threads_, enable_checks_);
}

inline Matrix Matrix::col(size_t idx) const {
    if (enable_checks_ && idx >= cols_) {
        throw std::out_of_range("Column index out of range");
    }
    std::vector<double> col_data(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        col_data[i] = (*this)(i, idx);
    }
    return Matrix(rows_, 1, col_data, num_threads_, enable_checks_);
}

// Statistical operations
inline double Matrix::min() const {
    if (!use_multithreading_ || num_threads_ <= 1) {
        return *std::min_element(data_.begin(), data_.end());
    }

    size_t chunk_size = (data_.size() + num_threads_ - 1) / num_threads_;
    std::vector<std::future<double>> futures;

    for (size_t i = 0; i < num_threads_; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, data_.size());
        if (start >= data_.size()) break;

        futures.push_back(std::async(std::launch::async, [this, start, end]() {
            return *std::min_element(data_.begin() + start, data_.begin() + end);
            }));
    }

    double min_val = std::numeric_limits<double>::max();
    for (auto& f : futures) {
        min_val = std::min(min_val, f.get());
    }
    return min_val;
}

inline double Matrix::max() const {
    if (!use_multithreading_ || num_threads_ <= 1) {
        return *std::max_element(data_.begin(), data_.end());
    }

    size_t chunk_size = (data_.size() + num_threads_ - 1) / num_threads_;
    std::vector<std::future<double>> futures;

    for (size_t i = 0; i < num_threads_; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, data_.size());
        if (start >= data_.size()) break;

        futures.push_back(std::async(std::launch::async, [this, start, end]() {
            return *std::max_element(data_.begin() + start, data_.begin() + end);
            }));
    }

    double max_val = std::numeric_limits<double>::lowest();
    for (auto& f : futures) {
        max_val = std::max(max_val, f.get());
    }
    return max_val;
}

inline double Matrix::sum() const {
    if (!use_multithreading_ || num_threads_ <= 1) {
        return std::accumulate(data_.begin(), data_.end(), 0.0);
    }

    size_t chunk_size = (data_.size() + num_threads_ - 1) / num_threads_;
    std::vector<std::future<double>> futures;

    for (size_t i = 0; i < num_threads_; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, data_.size());
        if (start >= data_.size()) break;

        futures.push_back(std::async(std::launch::async, [this, start, end]() {
            return std::accumulate(data_.begin() + start, data_.begin() + end, 0.0);
            }));
    }

    double total = 0.0;
    for (auto& f : futures) {
        total += f.get();
    }
    return total;
}

inline double Matrix::mean() const {
    return sum() / static_cast<double>(data_.size());
}

inline double Matrix::variance() const {
    double m = mean();
    double var_sum = 0.0;

    if (!use_multithreading_ || num_threads_ <= 1) {
        for (const auto& val : data_) {
            double diff = val - m;
            var_sum += diff * diff;
        }
    }
    else {
        size_t chunk_size = (data_.size() + num_threads_ - 1) / num_threads_;
        std::vector<std::future<double>> futures;

        for (size_t i = 0; i < num_threads_; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, data_.size());
            if (start >= data_.size()) break;

            futures.push_back(std::async(std::launch::async, [this, start, end, m]() {
                double local_sum = 0.0;
                for (size_t j = start; j < end; ++j) {
                    double diff = data_[j] - m;
                    local_sum += diff * diff;
                }
                return local_sum;
                }));
        }

        for (auto& f : futures) {
            var_sum += f.get();
        }
    }

    return var_sum / static_cast<double>(data_.size());
}

inline double Matrix::std_dev() const {
    return std::sqrt(variance());
}

// Element-wise math functions
inline Matrix Matrix::sqrt() const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::sqrt(data_[i]);
        }
        });

    if (enable_checks_) result.check_nan_inf();
    return result;
}

inline Matrix Matrix::exp() const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::exp(data_[i]);
        }
        });

    if (enable_checks_) result.check_nan_inf();
    return result;
}

inline Matrix Matrix::log() const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::log(data_[i]);
        }
        });

    if (enable_checks_) result.check_nan_inf();
    return result;
}

inline Matrix Matrix::pow(double exponent) const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::pow(data_[i], exponent);
        }
        });

    if (enable_checks_) result.check_nan_inf();
    return result;
}

inline Matrix Matrix::abs() const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::abs(data_[i]);
        }
        });

    return result;
}

inline Matrix Matrix::sin() const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::sin(data_[i]);
        }
        });

    return result;
}

inline Matrix Matrix::cos() const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::cos(data_[i]);
        }
        });

    return result;
}

inline Matrix Matrix::tan() const {
    Matrix result(rows_, cols_, 0.0, num_threads_, enable_checks_);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::tan(data_[i]);
        }
        });

    if (enable_checks_) result.check_nan_inf();
    return result;
}

// Utility functions
inline void Matrix::fill(double value) {
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        std::fill(data_.begin() + start, data_.begin() + end, value);
        });
}

inline Matrix Matrix::reshape(size_t new_rows, size_t new_cols) const {
    if (enable_checks_ && new_rows * new_cols != data_.size()) {
        throw std::invalid_argument("New dimensions must match total size");
    }
    return Matrix(new_rows, new_cols, data_, num_threads_, enable_checks_);
}

inline const std::vector<double>& Matrix::get_data() const { return data_; }
inline std::vector<double>& Matrix::get_data() { return data_; }

#endif // MATRIX_HPP