#ifndef MATRIX_CLASS_TEST_HPP
#define MATRIX_CLASS_TEST_HPP

#include "matrix.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>

class MatrixTest {
private:
    static constexpr double EPSILON = 1e-9;
    int passed_tests = 0;
    int failed_tests = 0;

    bool are_equal(double a, double b) const {
        return std::abs(a - b) < EPSILON;
    }

    bool are_matrices_equal(const Matrix& a, const Matrix& b) const {
        if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
        for (size_t i = 0; i < a.rows(); ++i) {
            for (size_t j = 0; j < a.cols(); ++j) {
                if (!are_equal(a(i, j), b(i, j))) return false;
            }
        }
        return true;
    }

    void report_test(const std::string& test_name, bool passed) {
        if (passed) {
            std::cout << "[PASS] " << test_name << std::endl;
            passed_tests++;
        }
        else {
            std::cout << "[FAIL] " << test_name << std::endl;
            failed_tests++;
        }
    }

    template<typename Func>
    double measure_time(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

public:
    void test_constructor() {
        Matrix m1(3, 4);
        report_test("Constructor with dimensions", m1.rows() == 3 && m1.cols() == 4);

        Matrix m2(2, 2, 5.0);
        bool all_five = true;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                if (m2(i, j) != 5.0) all_five = false;
            }
        }
        report_test("Constructor with init value", all_five);

        std::vector<double> data = { 1, 2, 3, 4 };
        Matrix m3(2, 2, data);
        report_test("Constructor with vector", m3(0, 0) == 1 && m3(1, 1) == 4);
    }

    void test_accessors() {
        Matrix m(3, 3, 0.0);
        m(1, 1) = 42.0;
        report_test("Element access", m(1, 1) == 42.0);
        report_test("Rows accessor", m.rows() == 3);
        report_test("Cols accessor", m.cols() == 3);
        report_test("Size accessor", m.size() == 9);
    }

    void test_addition() {
        std::vector<double> data1 = { 1, 2, 3, 4 };
        std::vector<double> data2 = { 5, 6, 7, 8 };
        Matrix m1(2, 2, data1);
        Matrix m2(2, 2, data2);
        Matrix result = m1 + m2;
        report_test("Matrix addition", result(0, 0) == 6 && result(1, 1) == 12);

        Matrix m3 = m1 + 10.0;
        report_test("Scalar addition", m3(0, 0) == 11 && m3(1, 1) == 14);
    }

    void test_subtraction() {
        std::vector<double> data1 = { 10, 20, 30, 40 };
        std::vector<double> data2 = { 1, 2, 3, 4 };
        Matrix m1(2, 2, data1);
        Matrix m2(2, 2, data2);
        Matrix result = m1 - m2;
        report_test("Matrix subtraction", result(0, 0) == 9 && result(1, 1) == 36);

        Matrix m3 = m1 - 5.0;
        report_test("Scalar subtraction", m3(0, 0) == 5 && m3(1, 1) == 35);
    }

    void test_multiplication() {
        std::vector<double> data1 = { 1, 2, 3, 4 };
        std::vector<double> data2 = { 2, 0, 1, 2 };
        Matrix m1(2, 2, data1);
        Matrix m2(2, 2, data2);
        Matrix result = m1 * m2;
        report_test("Matrix multiplication", result(0, 0) == 4 && result(1, 1) == 8);

        Matrix m3 = m1 * 2.0;
        report_test("Scalar multiplication", m3(0, 0) == 2 && m3(1, 1) == 8);
    }

    void test_element_wise_ops() {
        std::vector<double> data1 = { 2, 4, 6, 8 };
        std::vector<double> data2 = { 2, 2, 3, 4 };
        Matrix m1(2, 2, data1);
        Matrix m2(2, 2, data2);

        Matrix mult = m1.element_wise_multiply(m2);
        report_test("Element-wise multiplication", mult(0, 0) == 4 && mult(1, 1) == 32);

        Matrix div = m1.element_wise_divide(m2);
        report_test("Element-wise division", div(0, 0) == 1 && div(1, 1) == 2);
    }

    void test_transpose() {
        std::vector<double> data = { 1, 2, 3, 4, 5, 6 };
        Matrix m(2, 3, data);
        Matrix t = m.transpose();
        report_test("Transpose dimensions", t.rows() == 3 && t.cols() == 2);
        report_test("Transpose values", t(0, 0) == 1 && t(2, 1) == 6);
    }

    void test_trace() {
        std::vector<double> data = { 1, 2, 3, 4 };
        Matrix m(2, 2, data);
        double tr = m.trace();
        report_test("Trace", are_equal(tr, 5.0));
    }

    void test_row_col_extraction() {
        std::vector<double> data = { 1, 2, 3, 4, 5, 6 };
        Matrix m(2, 3, data);

        Matrix r = m.row(1);
        report_test("Row extraction", r.rows() == 1 && r.cols() == 3 && r(0, 2) == 6);

        Matrix c = m.col(1);
        report_test("Column extraction", c.rows() == 2 && c.cols() == 1 && c(1, 0) == 5);
    }

    void test_sub_matrix_extraction() {
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        Matrix m(3, 3, data);

        Matrix m1 = m.sub_matrix(1, 2, 1, 2);
        Matrix m1_real(2, 2, { 5, 6, 8, 9});
        //Matrix m1_real(3, 2, { 1, 2, 4, 5, 8, 9 });
        report_test("Submatrix extraction", m1.rows() == m1_real.rows() && m1.cols() == m1_real.cols());
    }

    void test_statistical_ops() {
        std::vector<double> data = { 1, 2, 3, 4, 5, 6 };
        Matrix m(2, 3, data);

        report_test("Min", are_equal(m.min(), 1.0));
        report_test("Max", are_equal(m.max(), 6.0));
        report_test("Sum", are_equal(m.sum(), 21.0));
        report_test("Mean", are_equal(m.mean(), 3.5));

        double var = m.variance();
        report_test("Variance", var > 2.9 && var < 3.0);

        double std = m.std_dev();
        report_test("Standard deviation", std > 1.7 && std < 1.8);
    }

    void test_math_functions() {
        std::vector<double> data = { 4, 9, 16, 25 };
        Matrix m(2, 2, data);

        Matrix sq = m.sqrt();
        report_test("Sqrt", are_equal(sq(0, 0), 2.0) && are_equal(sq(1, 1), 5.0));

        Matrix ab = (m * -1.0).abs();
        report_test("Abs", are_equal(ab(0, 0), 4.0) && are_equal(ab(1, 1), 25.0));

        Matrix p = m.pow(2);
        report_test("Pow", are_equal(p(0, 0), 16.0) && are_equal(p(1, 1), 625.0));
        double M_PI = 3.14159265358979323846;
        std::vector<double> data2 = { 0, M_PI / 2, M_PI, 3 * M_PI / 2 };
        Matrix m2(2, 2, data2);
        Matrix s = m2.sin();
        report_test("Sin", are_equal(s(0, 0), 0.0) && are_equal(s(0, 1), 1.0));

        Matrix c = m2.cos();
        report_test("Cos", are_equal(c(0, 0), 1.0) && are_equal(c(0, 1), 0.0));
    }

    void test_utility_functions() {
        Matrix m(3, 3);
        m.fill(7.0);
        bool all_seven = true;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (m(i, j) != 7.0) all_seven = false;
            }
        }
        report_test("Fill", all_seven);

        std::vector<double> data(12, 1.0);
        Matrix m2(3, 4, data);
        Matrix reshaped = m2.reshape(4, 3);
        report_test("Reshape", reshaped.rows() == 4 && reshaped.cols() == 3);
    }

    void test_in_place_ops() {
        std::vector<double> data = { 1, 2, 3, 4 };
        Matrix m1(2, 2, data);
        Matrix m2(2, 2, data);

        m1 += m2;
        report_test("In-place addition", m1(0, 0) == 2 && m1(1, 1) == 8);

        m1 -= m2;
        report_test("In-place subtraction", m1(0, 0) == 1 && m1(1, 1) == 4);

        m1 *= 3.0;
        report_test("In-place multiplication", m1(0, 0) == 3 && m1(1, 1) == 12);

        m1 /= 3.0;
        report_test("In-place division", m1(0, 0) == 1 && m1(1, 1) == 4);
    }

    void test_checks() {
        Matrix m1(2, 2, 0.0);
        Matrix m2(3, 3, 0.0);

        bool caught_dimension_error = false;
        try {
            Matrix result = m1 + m2;
        }
        catch (const std::invalid_argument&) {
            caught_dimension_error = true;
        }
        report_test("Dimension validation", caught_dimension_error);

        bool caught_div_zero = false;
        try {
            Matrix m3 = m1 / 0.0;
        }
        catch (const std::runtime_error&) {
            caught_div_zero = true;
        }
        report_test("Division by zero check", caught_div_zero);
    }

    void performance_comparison(size_t size = 500) {
        std::cout << "\n=== Performance Comparison (Matrix size: " << size << "x" << size << ") ===" << std::endl;

        std::vector<double> data1(size * size);
        std::vector<double> data2(size * size);
        for (size_t i = 0; i < size * size; ++i) {
            data1[i] = static_cast<double>(i % 100) / 10.0;
            data2[i] = static_cast<double>((i + 50) % 100) / 10.0;
        }

        // Test different operations
        auto test_operation = [&](const std::string& op_name, auto&& operation) {

            MATRIX_CONFIGS::ENABLE_MULTITHREADING = false;
            Matrix m1_single(size, size, data1);
            Matrix m2_single(size, size, data2);
            double time_single = measure_time([&]() { operation(m1_single, m2_single); });

            MATRIX_CONFIGS::ENABLE_MULTITHREADING = true;
            size_t num_threads = MATRIX_CONFIGS::NUM_THREADS;
            Matrix m1_multi(size, size);
            Matrix m2_multi(size, size);
            double time_multi = measure_time([&]() { operation(m1_multi, m2_multi); });

            MATRIX_CONFIGS::ENABLE_MULTITHREADING = false;
            double speedup = time_single / time_multi;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << op_name << ":\n";
            std::cout << "  Single-thread: " << time_single << " ms\n";
            std::cout << "  Multi-thread (" << num_threads << " threads): " << time_multi << " ms\n";
            std::cout << "  Speedup: " << speedup << "x\n" << std::endl;
            };

        test_operation("Addition", [](Matrix& a, Matrix& b) { auto r = a + b; });
        test_operation("Transpose", [](Matrix& a, Matrix& b) { auto r = a.transpose(); });
        test_operation("Element-wise sqrt", [](Matrix& a, Matrix& b) { auto r = a.sqrt(); });
        test_operation("Mean calculation", [](Matrix& a, Matrix& b) { auto r = a.mean(); });
        test_operation("Multiplication", [](Matrix& a, Matrix& b) { auto r = a * b; });
        
    }

    void run_all_tests() {
        MATRIX_CONFIGS::ENABLE_CHECKS = true;
        MATRIX_CONFIGS::NUM_THREADS = 4;
        std::cout << "=== Running Matrix Class Tests ===" << std::endl;

        test_constructor();
        test_accessors();
        test_addition();
        test_subtraction();
        test_multiplication();
        test_element_wise_ops();
        test_transpose();
        test_trace();
        test_row_col_extraction();
        test_sub_matrix_extraction();
        test_statistical_ops();
        test_math_functions();
        test_utility_functions();
        test_in_place_ops();
        test_checks();

        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Passed: " << passed_tests << std::endl;
        std::cout << "Failed: " << failed_tests << std::endl;
        std::cout << "Total: " << (passed_tests + failed_tests) << std::endl;

        if (failed_tests == 0) {
            std::cout << "\nAll tests passed!" << std::endl;
        }

        MATRIX_CONFIGS::ENABLE_CHECKS = false;
        MATRIX_CONFIGS::NUM_THREADS = 1;
    }
};

#endif // MATRIX_CLASS_TEST_HPP