cmake_minimum_required(VERSION 3.25)
project(cvtf_demo
    VERSION 1.0
    DESCRIPTION "Example OpenCV and Tesseract project."
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(cvtf_demo)

target_sources(cvtf_demo
    PRIVATE
        src/main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/include/config.hpp)

target_compile_features(cvtf_demo
    PRIVATE
        cxx_std_20)

target_link_libraries(cvtf_demo
    PRIVATE
        ${OpenCV_LIBS})

set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/config.hpp
    @ONLY)

target_include_directories(cvtf_demo
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}/include)

if(MSVC)
    target_compile_options(cvtf_demo PRIVATE /W4)
else()
    target_compile_options(cvtf_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NOT MSVC)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json)
endif()
