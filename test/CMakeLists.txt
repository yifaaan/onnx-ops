add_executable(test_scatter_elements
    test_scatter_elements.cpp
)

target_include_directories(test_scatter_elements
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_scatter_elements
    PRIVATE Catch2::Catch2 Catch2::Catch2WithMain Eigen3::Eigen
) 