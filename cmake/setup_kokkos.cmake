if (NOT KOKKOS_ROOT)
    set(KOKKOS_ROOT "")
    set(KOKKOS_ROOT "$ENV{KOKKOS_ROOT}")
endif()
message(STATUS "Kokkos root ${KOKKOS_ROOT}")
find_package(Kokkos REQUIRED PATHS ${KOKKOS_ROOT}/lib/cmake/Kokkos ${KOKKOS_ROOT}/lib64/cmake/Kokkos)

option(SKL_ENABLE_CUDA  "Enable CUDA device support" OFF) 
option(SKL_ENABLE_HIP   "Enable HIP device support"  OFF)
option(SKL_ENABLE_OMP   "Enable OpenMP threading support" OFF)
option(SKL_ENABLE_SERIAL "Enable serial execution of ParallelLoops" OFF)

if( (NOT SKL_ENABLE_CUDA)  AND (NOT SKL_ENABLE_HIP) AND (NOT SKL_ENABLE_OMP) )
    message(STATUS "No backend selectend, enabling serial execution")
    set(SKL_ENABLE_SERIAL ON)
endif() 

if( SKL_ENABLE_CUDA AND (NOT Kokkos_ENABLE_CUDA))
    message(FATAL_ERROR "SKL configured with CUDA support but Kokkos does not support CUDA backend.")
endif()

if( SKL_ENABLE_HIP AND (NOT Kokkos_ENABLE_HIP))
    message(FATAL_ERROR "SKL configured with HIP support but Kokkos does not support HIP backend.")
endif()

if( SKL_ENABLE_OMP AND (NOT Kokkos_ENABLE_OPENMP))
    message(FATAL_ERROR "SKL configured with OMP support but Kokkos does not support OMP backend.")
endif()

if( SKL_ENABLE_SERIAL AND (NOT Kokkos_ENABLE_SERIAL))
    message(FATAL_ERROR "SKL configured with SERIAL support but Kokkos does not support SERIAL backend.")
endif()

if( SKL_ENABLE_OMP )
    find_package(OpenMP REQUIRED)
endif()
