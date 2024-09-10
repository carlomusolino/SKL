if (NOT KOKKOSKERNELS_ROOT)
    set(KOKKOSKERNELS_ROOT "")
    set(KOKKOSKERNELS_ROOT "$ENV{KOKKOSKERNELS_ROOT}")
endif()
message(STATUS "KokkosKernels root ${KOKKOSKERNELS_ROOT}")
find_package(KokkosKernels REQUIRED PATHS ${KOKKOSKERNELS_ROOT}/lib/cmake/KokkosKernels ${KOKKOSKERNELS_ROOT}/lib64/cmake/KokkosKernels)


