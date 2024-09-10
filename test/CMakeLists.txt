add_executable(test_sacado_functionality test_sacado_functionality.cc)
target_include_directories(test_sacado_functionality PRIVATE "${HEADER_DIR}" "${CMAKE_BINARY_DIR}")
target_link_libraries(test_sacado_functionality PRIVATE Trilinos::Trilinos MPI::MPI_CXX)


add_executable(test_sacado_1d_solver 1d_spectral_solver.cc)
target_include_directories(test_sacado_1d_solver PRIVATE "${HEADER_DIR}" "${CMAKE_BINARY_DIR}")
target_link_libraries(test_sacado_1d_solver PRIVATE Trilinos::Trilinos MPI::MPI_CXX Kokkos::kokkos)

add_executable(test_jvp test_jvp.cc)
target_include_directories(test_jvp PRIVATE "${HEADER_DIR}" "${CMAKE_BINARY_DIR}")
target_link_libraries(test_jvp PRIVATE Trilinos::Trilinos MPI::MPI_CXX Kokkos::kokkos)


add_executable(test_kokkos_sfad tets_kokkos_gmres_solver_1d.cc)
target_include_directories(test_kokkos_sfad PRIVATE "${HEADER_DIR}" "${CMAKE_BINARY_DIR}")
target_link_libraries(test_kokkos_sfad PRIVATE Trilinos::Trilinos MPI::MPI_CXX Kokkos::kokkos)


add_executable(test_blas test_blas_implementation.cc)
target_include_directories(test_blas PRIVATE "${HEADER_DIR}" "${CMAKE_BINARY_DIR}")
target_link_libraries(test_blas PRIVATE Trilinos::Trilinos MPI::MPI_CXX Kokkos::kokkos KokkosKernels::kokkoskernels)