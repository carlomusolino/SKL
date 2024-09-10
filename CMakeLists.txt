cmake_minimum_required(VERSION 3.14)

project(GRACE_ID VERSION 0.0 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(prohibit_in_source_build)

configure_file("${CMAKE_SOURCE_DIR}/include/grace_id/grace_id_config.h.in" grace_id_config.h)

set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include/")

option(GRACE_ID_USE_FP64 "Use double precision arithmetic" ON)

include(setup_mpi)
include(setup_kokkos)
include(setup_kokkoskernels)
include(setup_yaml)
include(setup_trilinois)

add_subdirectory(src)

option(GRACE_ID_ENABLE_TESTING "Enable tests." ON)
if( GRACE_ID_ENABLE_TESTING )
    include(setup_catch2)
    message(STATUS "Testing enabled.")
    add_subdirectory(test)
endif() 