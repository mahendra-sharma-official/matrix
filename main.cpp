#include "matrix_test.hpp"

int main() {
    MatrixTest tester;

    
    THREAD_CONFIGS::NUM_THREADS = 2;
    THREAD_CONFIGS::ENABLE_MULTITHREADING = true;
    GLOBAL_THREAD_POOL.resize(NUM_THREADS);
    // Run all correctness tests
    tester.run_all_tests();

    THREAD_CONFIGS::NUM_THREADS = 2;
    THREAD_CONFIGS::ENABLE_MULTITHREADING = true;

    // Run performance benchmarks (you can adjust size)
    tester.performance_comparison(600);  // or 1000, 2000, etc.

    return 0;
}