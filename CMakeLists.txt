cmake_minimum_required(VERSION 3.14)

project(SPEKTRALIB VERSION 0.0 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(prohibit_in_source_build)

configure_file("${CMAKE_SOURCE_DIR}/include/SKL/SKL_config.h.in" SKL_config.h)

set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include/")

option(SpeKtraLib_USE_FP64 "Use double precision arithmetic" ON)

include(setup_mpi)
include(setup_kokkos)
include(setup_kokkoskernels)
include(setup_yaml)
include(setup_trilinois)

add_subdirectory(src)

option(SpeKtraLib_ENABLE_TESTING "Enable tests." ON)
if( SpeKtraLib_ENABLE_TESTING )
    include(setup_catch2)
    message(STATUS "Testing enabled.")
    add_subdirectory(test)
endif() 