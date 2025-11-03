#include "matrix_test.hpp"

int main() {
    MatrixTest tester;

    MATRIX_CONFIGS::NUM_THREADS = 4;
    MATRIX_CONFIGS::ENABLE_MULTITHREADING = true;

    // Run all correctness tests
    tester.run_all_tests();

    MATRIX_CONFIGS::NUM_THREADS = 4;
    MATRIX_CONFIGS::ENABLE_MULTITHREADING = true;

    // Run performance benchmarks (you can adjust size)
    tester.performance_comparison(600);  // or 1000, 2000, etc.

    return 0;
}