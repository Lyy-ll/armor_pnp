cmake_minimum_required(VERSION 3.16)

project(armor_pnp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(armor_pnp
    main.cpp
    armor_detector.cpp
    pnp_solver.cpp
)

target_include_directories(armor_pnp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(armor_pnp PRIVATE ${OpenCV_LIBS})
