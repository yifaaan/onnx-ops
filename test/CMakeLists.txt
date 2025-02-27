add_executable(test_scatter_elements
    test_scatter_elements.cpp
)

add_executable(test_compress
    test_compress.cpp
)

add_executable(test_gather_elements
    test_gather_elements.cpp
)

add_executable(test_gather_nd
    test_gather_nd.cpp
)

target_include_directories(test_gather_elements
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_scatter_elements
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_compress
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_gather_nd
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_scatter_elements
    PRIVATE Catch2::Catch2 Catch2::Catch2WithMain Eigen3::Eigen
) 

target_link_libraries(test_compress
    PRIVATE Catch2::Catch2 Catch2::Catch2WithMain
) 

target_link_libraries(test_gather_elements
    PRIVATE Catch2::Catch2 Catch2::Catch2WithMain
)

target_link_libraries(test_gather_nd
    PRIVATE Catch2::Catch2 Catch2::Catch2WithMain
)
