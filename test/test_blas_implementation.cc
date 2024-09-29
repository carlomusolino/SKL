#include <SKL_config.h>

#include <SKL/utils/linalg.hh>
#include <SKL/utils/types.hh> 

#include <Sacado.hpp>
#include <vector>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp> 

#include <cassert> 

int main() {
    using namespace skl ; 
    constexpr size_t n_der = 1  ;   
    constexpr size_t m     = 10 ;

    Kokkos::initialize() ;
    {
        Kokkos::View<sfad_t<n_der>*, Kokkos::DefaultExecutionSpace>
            a_fad("A_fad", m, n_der+1), b_fad("B_fad", m, n_der+1) ;
        Kokkos::View<GRACE_REAL*, Kokkos::DefaultExecutionSpace> 
            a("A", m), b("B", m) ; 

        Kokkos::parallel_for("fill", m, 
            KOKKOS_LAMBDA( int i) 
        {
            a_fad(i) = sfad_t<n_der>(1.) ; 
            b_fad(i) = sfad_t<n_der>(2.) ;
            a(i)     = 1. ; 
            b(i)     = 2. ;
        })  ; 

        auto a_fad_norm = utils::linalg::nrm2(a_fad) ; 
        auto a_norm     = utils::linalg::nrm2(a)     ; 

        CHECK_THAT( 
            Kokkos::fabs(a_fad_norm - a_norm ),
            Catch::Matchers::WithinAbs(0, 1e-10 ) ) ; 
        CHECK_THAT( 
            Kokkos::fabs(a_norm - Kokkos::sqrt(10) ),
            Catch::Matchers::WithinAbs(0, 1e-10 ) ) ; 

        auto ab_fad  = utils::linalg::dot(a_fad,b_fad) ; 
        auto a_fad_b = utils::linalg::dot(a_fad,b)     ; 
        auto a_b_fad  = utils::linalg::dot(a,b_fad)     ; 
        auto ab      = utils::linalg::dot(a,b)         ;
        
        CHECK_THAT( 
            Kokkos::fabs(ab_fad - 20. ),
            Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        CHECK_THAT( 
            Kokkos::fabs(ab_fad - a_fad_b ),
            Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        CHECK_THAT( 
            Kokkos::fabs(ab_fad - a_b_fad ),
            Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        CHECK_THAT( 
            Kokkos::fabs(ab_fad - ab ),
            Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;


        // Now test scal 
        double alpha{2} ; 
        sfad_t<n_der> alpha_fad{2.} ; 
        Kokkos::View<sfad_t<n_der>*, Kokkos::DefaultExecutionSpace>
            y_fad("Y_fad", m, n_der+1) ; 
        Kokkos::View<GRACE_REAL*, Kokkos::DefaultExecutionSpace> 
            y("Y", m) ; 
        
        // First: try all fad 
        utils::linalg::scal(y_fad, alpha_fad, a_fad) ; 
        auto h_y_fad = Kokkos::create_mirror_view(y_fad) ; 
        Kokkos::deep_copy(h_y_fad,y_fad) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y_fad(i).val() - 2.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try having alpha as double 
        utils::linalg::scal(y_fad, alpha, a_fad) ; 
        Kokkos::deep_copy(h_y_fad,y_fad) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y_fad(i).val() - 2.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try having x as View<double*> 
        utils::linalg::scal(y_fad, alpha_fad, a) ; 
        Kokkos::deep_copy(h_y_fad,y_fad) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y_fad(i).val() - 2.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try having x sas View<double*> and alpha as double
        utils::linalg::scal(y_fad, alpha, a) ; 
        Kokkos::deep_copy(h_y_fad,y_fad) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y_fad(i).val() - 2.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ; 
        }
        // Then: try all scalar types
        utils::linalg::scal(y, alpha, a) ; 
        auto h_y = Kokkos::create_mirror_view(y) ; 
        Kokkos::deep_copy(h_y,y) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y(i) - 2.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ; 
        }
        // Then: try alpha fad scalar types
        utils::linalg::scal(y, alpha_fad, a) ; 
        Kokkos::deep_copy(h_y,y) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y(i) - 2.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try x fad scalar types
        utils::linalg::scal(y, alpha, a_fad) ; 
        Kokkos::deep_copy(h_y,y) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y(i) - 2.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try alpha fad scalar types
        utils::linalg::scal(y, alpha_fad, a_fad) ; 
        Kokkos::deep_copy(h_y,y) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y(i) - 2.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        
        // Test axpy 
        // First: try all fad 
        utils::linalg::axpy(alpha_fad, a_fad, y_fad) ; 
        Kokkos::deep_copy(h_y_fad,y_fad) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y_fad(i).val() - 4.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try having alpha as double 
        utils::linalg::axpy(alpha, a_fad, y_fad) ; 
        Kokkos::deep_copy(h_y_fad,y_fad) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y_fad(i).val() - 4.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try having x as View<double*> 
        utils::linalg::axpy(alpha_fad, a, y_fad) ; 
        Kokkos::deep_copy(h_y_fad,y_fad) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y_fad(i).val() - 4.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try having x sas View<double*> and alpha as double
        utils::linalg::axpy(alpha, a, y_fad) ; 
        Kokkos::deep_copy(h_y_fad,y_fad) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y_fad(i).val() - 4.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try all scalar types
        utils::linalg::axpy(alpha, a, y) ; 
        Kokkos::deep_copy(h_y,y) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y(i) - 4.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try alpha fad scalar types
        utils::linalg::axpy(alpha_fad, a, y) ; 
        Kokkos::deep_copy(h_y,y) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y(i) - 4.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try x fad scalar types
        utils::linalg::axpy(alpha, a_fad, y) ; 
        Kokkos::deep_copy(h_y,y) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y(i) - 4.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then: try alpha fad scalar types
        utils::linalg::axpy(alpha_fad, a_fad, y) ; 
        Kokkos::deep_copy(h_y,y) ; 
        for( int i=0; i<m; ++i) {
            CHECK_THAT( 
                Kokkos::fabs(h_y(i) - 4.),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }

        // Check the BLAS-3 trsm routine
        static constexpr size_t m_mat = 3 ; 

        const char side [] = "L" ; 
        const char uplo [] = "U" ; 
        const char trans [] = "N" ; 
        const char diag [] = "N"  ; 
        
        Kokkos::View<sfad_t<n_der>**, Kokkos::DefaultExecutionSpace>
            A_fad("fad_trsm_A", m_mat, m_mat, n_der+1), B_fad("fad_trsm_B", m_mat,1, n_der+1) ; 
        Kokkos::View<sfad_t<n_der>*, Kokkos::DefaultExecutionSpace>
            B_oned_fad("fad_trsm_B_oned", m_mat, n_der+1) ; 
        Kokkos::View<GRACE_REAL**, Kokkos::DefaultExecutionSpace> 
            A("trsm_A", m_mat,m_mat), B("trsm_B", m_mat,m_mat) ; 
        Kokkos::View<GRACE_REAL*, Kokkos::DefaultExecutionSpace> 
            B_oned("trsm_B_oned", m_mat) ; 

        auto h_B_fad = Kokkos::create_mirror_view(B_fad) ;
        auto h_A_fad = Kokkos::create_mirror_view(A_fad) ;
        auto h_B_oned_fad = Kokkos::create_mirror_view(B_oned_fad) ;

        auto h_B      = Kokkos::create_mirror_view(B) ;
        auto h_A      = Kokkos::create_mirror_view(A) ;
        auto h_B_oned = Kokkos::create_mirror_view(B_oned) ;

        h_A_fad(0,0) = 2; h_A_fad(0,1) = 3; h_A_fad(0,2) = 1;  
        h_A_fad(1,1) = 1; h_A_fad(1,2) = 4; h_A_fad(2,2) = 2;  

        h_A(0,0) = 2; h_A(0,1) = 3; h_A(0,2) = 1;  
        h_A(1,1) = 1; h_A(1,2) = 4; h_A(2,2) = 2;  

        h_B_fad(0,0) = 5 ; h_B_fad(1,0) = 4 ; h_B_fad(2,0) = 2 ; 
        h_B_oned_fad(0) = 5 ; h_B_oned_fad(1) = 4 ; h_B_oned_fad(2) = 2 ; 

        h_B(0,0) = 5 ; h_B(1,0) = 4 ; h_B(2,0) = 2 ; 
        h_B_oned(0) = 5 ; h_B_oned(1) = 4 ; h_B_oned(2) = 2 ; 

        Kokkos::deep_copy(A_fad, h_A_fad) ;
        Kokkos::deep_copy(A, h_A) ; 
        Kokkos::deep_copy(B_fad, h_B_fad) ; 
        Kokkos::deep_copy(B, h_B) ; 
        Kokkos::deep_copy(B_oned_fad, h_B_oned_fad) ; 
        Kokkos::deep_copy(B_oned, h_B_oned) ;

        std::vector<double> sol{4,0,2} ; 
        // First try all fad 
        utils::linalg::trsm(side,uplo,trans,diag, alpha_fad, A_fad, B_fad) ; 
        Kokkos::deep_copy(h_B_fad, B_fad) ; 
        for(int ii=0; ii<m_mat; ++ii) {
            CHECK_THAT( 
                Kokkos::fabs(sol[ii] - h_B_fad(ii,0).val()),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Then try all fad with oned B
        utils::linalg::trsm(side,uplo,trans,diag, alpha_fad, A_fad, B_oned_fad) ; 
        Kokkos::deep_copy(h_B_oned_fad, B_oned_fad) ; 
        for(int ii=0; ii<m_mat; ++ii) {
            CHECK_THAT( 
                Kokkos::fabs(sol[ii] - h_B_oned_fad(ii).val()),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // Now try A non fad
        utils::linalg::trsm(side,uplo,trans,diag, alpha_fad, A, B_fad) ; 
        Kokkos::deep_copy(h_B_fad, B_fad) ; 
        for(int ii=0; ii<m_mat; ++ii) {
            CHECK_THAT( 
                Kokkos::fabs(sol[ii] - h_B_oned_fad(ii).val()),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // And with oned B
        utils::linalg::trsm(side,uplo,trans,diag, alpha_fad, A, B_oned_fad) ; 
        Kokkos::deep_copy(h_B_oned_fad, B_oned_fad) ; 
        for(int ii=0; ii<m_mat; ++ii) {
            CHECK_THAT( 
                Kokkos::fabs(sol[ii] - h_B_oned_fad(ii).val()),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }

        // Now try B non fad
        utils::linalg::trsm(side,uplo,trans,diag, alpha, A, B) ; 
        Kokkos::deep_copy(h_B, B) ; 
        for(int ii=0; ii<m_mat; ++ii) {
            assert(Kokkos::fabs(sol[ii] - h_B(ii,0).val()) < 1e-10 ) ;
        }
        // And with oned B
        utils::linalg::trsm(side,uplo,trans,diag, alpha, A, B_oned) ; 
        Kokkos::deep_copy(h_B_oned, B_oned) ; 
        for(int ii=0; ii<m_mat; ++ii) {
            CHECK_THAT( 
                Kokkos::fabs(sol[ii] - h_B_oned(ii).val()),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }

        // Go back to A_fad
        utils::linalg::trsm(side,uplo,trans,diag, alpha, A_fad, B) ; 
        Kokkos::deep_copy(h_B, B) ; 
        for(int ii=0; ii<m_mat; ++ii) {
            CHECK_THAT( 
                Kokkos::fabs(sol[ii] - h_B(ii,0).val()),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }
        // And with oned B
        utils::linalg::trsm(side,uplo,trans,diag, alpha, A_fad, B_oned) ; 
        Kokkos::deep_copy(h_B_oned, B_oned) ; 
        for(int ii=0; ii<m_mat; ++ii) {
            CHECK_THAT( 
                Kokkos::fabs(sol[ii] - h_B_oned(ii).val()),
                Catch::Matchers::WithinAbs(0, 1e-10 ) ) ;
        }

    }
    Kokkos::finalize() ; 

}