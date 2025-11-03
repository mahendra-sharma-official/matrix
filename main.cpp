#include "matrix_test.hpp"

int main() {
    MatrixTest tester;

    // Run all correctness tests
    tester.run_all_tests();

    // Run performance benchmarks (you can adjust size)
    tester.performance_comparison(1000);  // or 1000, 2000, etc.

    return 0;
}