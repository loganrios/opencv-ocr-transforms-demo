cmake_minimum_required(VERSION 3.25)
project(tsrt_demo
    VERSION 1.0
    DESCRIPTION "Example OpenCV and Tesseract project."
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(tsrt_demo)

target_sources(tsrt_demo
    PRIVATE
        src/main.cpp)

target_compile_features(tsrt_demo
    PRIVATE
        cxx_std_20)

target_link_libraries(tsrt_demo
    PRIVATE
        ${OpenCV_LIBS})

target_include_directories(tsrt_demo
    PRIVATE
        ${OpenCV_INCLUDE_DIRS})

if(MSVC)
    target_compile_options(tsrt_demo PRIVATE /W4)
else()
    target_compile_options(tsrt_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
