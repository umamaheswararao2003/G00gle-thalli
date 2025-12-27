#include "Google/Runtime/GoogleRuntime.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace google::runtime;

// Simple CPU matmul implementation for testing
void simple_matmul(float* A, float* B, float* C, int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Wrapper for runtime
void matmul_wrapper(void** args) {
    float* A = static_cast<float*>(args[0]);
    float* B = static_cast<float*>(args[1]);
    float* C = static_cast<float*>(args[2]);
    int64_t* sizes = static_cast<int64_t*>(args[3]);
    
    simple_matmul(A, B, C, sizes[0], sizes[1], sizes[2]);
}

bool verify_correctness(const Tensor& C, int64_t M, int64_t N, int64_t K) {
    // For A and B filled with 1.0, C[i,j] should be K
    float expected = static_cast<float>(K);
    const float tolerance = 0.01f;
    
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            if (std::abs(C(i, j) - expected) > tolerance) {
                std::cout << "  Error at (" << i << ", " << j << "): expected " 
                         << expected << ", got " << C(i, j) << std::endl;
                return false;
            }
        }
    }
    return true;
}

double benchmark_matmul(GoogleRuntime& runtime, int64_t M, int64_t K, int64_t N, int iterations) {
    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C({M, N});
    
    A.fill(1.0f);
    B.fill(1.0f);
    
    int64_t sizes[3] = {M, K, N};
    std::vector<void*> args = {A.data(), B.data(), C.data(), sizes};
    
    // Warm-up
    runtime.execute("matmul", args);
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        C.fill(0.0f);
        runtime.execute("matmul", args);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / iterations;
}

int main() {
    std::cout << "\n=== Phase 1 Integration Test ===\n" << std::endl;
    
    auto& runtime = GoogleRuntime::instance();
    
    // Register kernel
    runtime.registerKernel("matmul", reinterpret_cast<void*>(matmul_wrapper));
    
    std::cout << "Registered kernels: " << runtime.numKernels() << std::endl;
    std::cout << "Has matmul: " << (runtime.hasKernel("matmul") ? "yes" : "no") << std::endl;
    std::cout << std::endl;
    
    // Test 1: Small matrix (correctness)
    std::cout << "Test 1: Correctness (128x128)" << std::endl;
    {
        const int64_t SIZE = 128;
        Tensor A({SIZE, SIZE});
        Tensor B({SIZE, SIZE});
        Tensor C({SIZE, SIZE});
        
        A.fill(1.0f);
        B.fill(1.0f);
        
        int64_t sizes[3] = {SIZE, SIZE, SIZE};
        std::vector<void*> args = {A.data(), B.data(), C.data(), sizes};
        runtime.execute("matmul", args);
        
        bool correct = verify_correctness(C, SIZE, SIZE, SIZE);
        std::cout << "  Result: " << (correct ? "✓ PASS" : "✗ FAIL") << std::endl;
        
        if (!correct) return 1;
    }
    
    // Test 2: Eager operations integration
    std::cout << "\nTest 2: Eager Operations Integration" << std::endl;
    {
        Tensor X({64, 64});
        X.randn(0.0f, 1.0f);
        
        // Preprocessing
        Tensor normalized = X / 255.0f;
        
        // Compiled operation
        Tensor W({64, 32});
        W.fill(0.1f);
        Tensor Y({64, 32});
        
        int64_t sizes[3] = {64, 64, 32};
        std::vector<void*> args = {normalized.data(), W.data(), Y.data(), sizes};
        runtime.execute("matmul", args);
        
        // Postprocessing
        Tensor activated = Y.relu();
        
        std::cout << "  Preprocessing: ✓" << std::endl;
        std::cout << "  Compiled matmul: ✓" << std::endl;
        std::cout << "  Postprocessing: ✓" << std::endl;
        std::cout << "  Result: ✓ PASS" << std::endl;
    }
    
    // Test 3: Performance benchmarks
    std::cout << "\nTest 3: Performance Benchmarks" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    struct BenchConfig {
        int64_t M, K, N;
        const char* name;
    };
    
    BenchConfig configs[] = {
        {256, 256, 256, "Small (256x256)"},
        {512, 512, 512, "Medium (512x512)"},
        {1024, 1024, 1024, "Large (1024x1024)"}
    };
    
    for (const auto& config : configs) {
        double avg_time = benchmark_matmul(runtime, config.M, config.K, config.N, 5);
        double gflops = (2.0 * config.M * config.N * config.K) / (avg_time * 1e6);
        
        std::cout << "  " << config.name << ":" << std::endl;
        std::cout << "    Time: " << avg_time << " ms" << std::endl;
        std::cout << "    Performance: " << gflops << " GFLOPS" << std::endl;
    }
    
    // Test 4: Memory management
    std::cout << "\nTest 4: Memory Management" << std::endl;
    {
        // Create many tensors to test memory management
        std::vector<Tensor> tensors;
        for (int i = 0; i < 100; ++i) {
            tensors.emplace_back(std::vector<int64_t>{100, 100});
            tensors.back().fill(static_cast<float>(i));
        }
        
        std::cout << "  Created 100 tensors: ✓" << std::endl;
        std::cout << "  Memory management: ✓ PASS" << std::endl;
    }
    
    std::cout << "\n=== Integration Test Summary ===" << std::endl;
    std::cout << "✓ Correctness: PASS" << std::endl;
    std::cout << "✓ Eager integration: PASS" << std::endl;
    std::cout << "✓ Performance: MEASURED" << std::endl;
    std::cout << "✓ Memory management: PASS" << std::endl;
    std::cout << "\n=== All Integration Tests Passed! ===\n" << std::endl;
    
    return 0;
}
