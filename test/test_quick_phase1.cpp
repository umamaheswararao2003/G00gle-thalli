#include "Google/Runtime/GoogleRuntime.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace google::runtime;

// Test counter
int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) \
    std::cout << "Testing " << name << "... "; \
    try {

#define END_TEST \
        std::cout << "✓ PASS" << std::endl; \
        tests_passed++; \
    } catch (const std::exception& e) { \
        std::cout << "✗ FAIL: " << e.what() << std::endl; \
        tests_failed++; \
    }

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error("Expected " + std::to_string(a) + " == " + std::to_string(b)); \
    }

#define ASSERT_NEAR(a, b, tol) \
    if (std::abs((a) - (b)) > (tol)) { \
        throw std::runtime_error("Expected " + std::to_string(a) + " ≈ " + std::to_string(b)); \
    }

#define ASSERT_TRUE(cond) \
    if (!(cond)) { \
        throw std::runtime_error("Expected condition to be true"); \
    }

// Mock kernel for testing
void mock_add_kernel(void** args) {
    float* a = static_cast<float*>(args[0]);
    float* b = static_cast<float*>(args[1]);
    float* c = static_cast<float*>(args[2]);
    int64_t size = *static_cast<int64_t*>(args[3]);
    
    for (int64_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

void test_runtime_singleton() {
    TEST("Runtime singleton")
    
    auto& runtime1 = GoogleRuntime::instance();
    auto& runtime2 = GoogleRuntime::instance();
    
    // Should be same instance
    ASSERT_TRUE(&runtime1 == &runtime2);
    
    END_TEST
}

void test_kernel_registration() {
    TEST("Kernel registration")
    
    auto& runtime = GoogleRuntime::instance();
    
    // Register a kernel
    runtime.registerKernel("test_kernel", reinterpret_cast<void*>(mock_add_kernel));
    
    // Check it's registered
    ASSERT_TRUE(runtime.hasKernel("test_kernel"));
    ASSERT_TRUE(!runtime.hasKernel("nonexistent"));
    
    END_TEST
}

void test_kernel_execution() {
    TEST("Kernel execution")
    
    auto& runtime = GoogleRuntime::instance();
    
    // Create simple arrays
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int64_t size = 4;
    
    // Execute kernel
    std::vector<void*> args = {a, b, c, &size};
    runtime.execute("test_kernel", args);
    
    // Verify results
    ASSERT_NEAR(c[0], 6.0f, 0.001f);
    ASSERT_NEAR(c[1], 8.0f, 0.001f);
    ASSERT_NEAR(c[2], 10.0f, 0.001f);
    ASSERT_NEAR(c[3], 12.0f, 0.001f);
    
    END_TEST
}

void test_tensor_creation() {
    TEST("Tensor creation")
    
    Tensor t({10, 20});
    
    ASSERT_EQ(t.ndim(), 2);
    ASSERT_EQ(t.rows(), 10);
    ASSERT_EQ(t.cols(), 20);
    ASSERT_EQ(t.size(), 200);
    
    END_TEST
}

void test_tensor_shape_strides() {
    TEST("Tensor shape and strides")
    
    Tensor t({4, 5});
    
    auto shape = t.shape();
    ASSERT_EQ(shape.size(), 2);
    ASSERT_EQ(shape[0], 4);
    ASSERT_EQ(shape[1], 5);
    
    auto strides = t.strides();
    ASSERT_EQ(strides.size(), 2);
    ASSERT_EQ(strides[0], 5);  // Row-major: stride for dim 0
    ASSERT_EQ(strides[1], 1);  // stride for dim 1
    
    END_TEST
}

void test_tensor_fill() {
    TEST("Tensor fill")
    
    Tensor t({3, 3});
    t.fill(42.0f);
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_NEAR(t(i, j), 42.0f, 0.001f);
        }
    }
    
    END_TEST
}

void test_tensor_scalar_division() {
    TEST("Tensor scalar division")
    
    Tensor t({2, 2});
    t.fill(10.0f);
    
    Tensor result = t / 2.0f;
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_NEAR(result(i, j), 5.0f, 0.001f);
        }
    }
    
    END_TEST
}

void test_tensor_scalar_multiplication() {
    TEST("Tensor scalar multiplication")
    
    Tensor t({2, 2});
    t.fill(3.0f);
    
    Tensor result = t * 4.0f;
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_NEAR(result(i, j), 12.0f, 0.001f);
        }
    }
    
    END_TEST
}

void test_tensor_addition() {
    TEST("Tensor element-wise addition")
    
    Tensor a({2, 2});
    Tensor b({2, 2});
    
    a.fill(1.0f);
    b.fill(2.0f);
    
    Tensor c = a + b;
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_NEAR(c(i, j), 3.0f, 0.001f);
        }
    }
    
    END_TEST
}

void test_tensor_subtraction() {
    TEST("Tensor element-wise subtraction")
    
    Tensor a({2, 2});
    Tensor b({2, 2});
    
    a.fill(5.0f);
    b.fill(3.0f);
    
    Tensor c = a - b;
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_NEAR(c(i, j), 2.0f, 0.001f);
        }
    }
    
    END_TEST
}

void test_tensor_multiplication() {
    TEST("Tensor element-wise multiplication")
    
    Tensor a({2, 2});
    Tensor b({2, 2});
    
    a.fill(3.0f);
    b.fill(4.0f);
    
    Tensor c = a * b;
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_NEAR(c(i, j), 12.0f, 0.001f);
        }
    }
    
    END_TEST
}

void test_tensor_relu() {
    TEST("Tensor ReLU activation")
    
    Tensor t({2, 2});
    t(0, 0) = -1.0f;
    t(0, 1) = 2.0f;
    t(1, 0) = -3.0f;
    t(1, 1) = 4.0f;
    
    Tensor result = t.relu();
    
    ASSERT_NEAR(result(0, 0), 0.0f, 0.001f);
    ASSERT_NEAR(result(0, 1), 2.0f, 0.001f);
    ASSERT_NEAR(result(1, 0), 0.0f, 0.001f);
    ASSERT_NEAR(result(1, 1), 4.0f, 0.001f);
    
    END_TEST
}

void test_tensor_sigmoid() {
    TEST("Tensor sigmoid activation")
    
    Tensor t({1, 1});
    t(0, 0) = 0.0f;
    
    Tensor result = t.sigmoid();
    
    // sigmoid(0) = 0.5
    ASSERT_NEAR(result(0, 0), 0.5f, 0.001f);
    
    END_TEST
}

void test_tensor_tanh() {
    TEST("Tensor tanh activation")
    
    Tensor t({1, 1});
    t(0, 0) = 0.0f;
    
    Tensor result = t.tanh();
    
    // tanh(0) = 0
    ASSERT_NEAR(result(0, 0), 0.0f, 0.001f);
    
    END_TEST
}

void test_tensor_randn() {
    TEST("Tensor random initialization")
    
    Tensor t({100, 100});
    t.randn(0.0f, 1.0f);
    
    // Check that values are different (not all zero)
    float sum = 0.0f;
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            sum += t(i, j);
        }
    }
    
    // With 10000 samples from N(0,1), sum should not be exactly 0
    ASSERT_TRUE(std::abs(sum) > 0.1f);
    
    END_TEST
}

void test_memory_alignment() {
    TEST("Memory alignment")
    
    auto& runtime = GoogleRuntime::instance();
    
    void* ptr = runtime.allocateAligned(1024, 64);
    
    // Check 64-byte alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    ASSERT_EQ(addr % 64, 0);
    
    runtime.deallocate(ptr);
    
    END_TEST
}

int main() {
    std::cout << "\n=== Phase 1 Quick Tests ===\n" << std::endl;
    
    // Runtime tests
    test_runtime_singleton();
    test_kernel_registration();
    test_kernel_execution();
    test_memory_alignment();
    
    // Tensor tests
    test_tensor_creation();
    test_tensor_shape_strides();
    test_tensor_fill();
    test_tensor_scalar_division();
    test_tensor_scalar_multiplication();
    test_tensor_addition();
    test_tensor_subtraction();
    test_tensor_multiplication();
    test_tensor_relu();
    test_tensor_sigmoid();
    test_tensor_tanh();
    test_tensor_randn();
    
    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    std::cout << "Total:  " << (tests_passed + tests_failed) << std::endl;
    
    if (tests_failed == 0) {
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed!" << std::endl;
        return 1;
    }
}
