# Add quick test executable
add_executable(test_quick_phase1
    test_quick_phase1.cpp
)

target_link_libraries(test_quick_phase1
    GoogleRuntime
)

target_include_directories(test_quick_phase1 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Add integration test executable
add_executable(test_integration_phase1
    test_integration_phase1.cpp
)

target_link_libraries(test_integration_phase1
    GoogleRuntime
)

target_include_directories(test_integration_phase1 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
