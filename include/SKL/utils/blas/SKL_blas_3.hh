/**
 * @file grace_blas_3.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-10
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *                                    
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *   
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *   
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */

#ifndef SKL_UTILS_BLAS_3_HH
#define SKL_UTILS_BLAS_3_HH

#include <SKL_config.h>

#include <SKL/utils/inline.h>
#include <SKL/utils/types.hh>

#include <Kokkos_Core.hpp>
#include <KokkosBlas3_trsm.hpp> 

#include <Sacado.hpp>


namespace utils { namespace linalg {

template< typename view_a_t 
        , typename view_b_t > 
void trsm(
    const char side[],
    const char uplo[],
    const char trans[],
    const char diag[],
    typename view_b_t::const_value_type& alpha,
    const view_a_t & A,
    const view_b_t & B ) 
{
    using scalar_a_t = typename view_a_t::const_value_type ; 
    using scalar_b_t = typename view_b_t::const_value_type ;

    constexpr size_t rank_a = view_a_t::rank() ; 
    constexpr size_t rank_b = view_b_t::rank() ; 
    static_assert( rank_a == 2 
             and ( rank_b == 2 or rank_b == 1), "trsm only supports rank-2 or 1 Views.") ; 
    


    if constexpr (  Sacado::IsFad<scalar_a_t>::value 
                and Sacado::IsFad<scalar_b_t>::value ) 
    {
        Kokkos::View<GRACE_REAL**, Kokkos::DefaultExecutionSpace>
            _A("trsm_A", A.extent(0), A.extent(1) ) ; 
        Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{A.extent(0),A.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                _A(i,j) = A(i,j).val() ;                 
            }) ;
        const GRACE_REAL _alpha = alpha.val() ; 

        if constexpr( rank_b == 1) {
            Kokkos::View<GRACE_REAL**, Kokkos::DefaultExecutionSpace>
                _B("trsm_B", B.extent(0), 1 ) ; 
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),1UL})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                _B(i,j) = B(i).val() ;                 
            }) ; 
            KokkosBlas::trsm(side,uplo,trans,diag,_alpha,_A,_B) ; 
            // copy data back
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),B.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                B(i) = _B(i,j) ;                 
            }) ;
        } else {
            Kokkos::View<GRACE_REAL**, Kokkos::DefaultExecutionSpace>
                _B("trsm_B", B.extent(0), B.extent(1) ) ;
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),B.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                _B(i,j) = B(i,j).val() ;                 
            }) ; 
            KokkosBlas::trsm(side,uplo,trans,diag,_alpha,_A,_B) ;
            #if 1
            // copy data back
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),B.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                B(i,j) = _B(i,j) ;                 
            }) ;
            #endif 
        }
    } else if constexpr ( Sacado::IsFad<scalar_a_t>::value  ) {
        Kokkos::View<GRACE_REAL**, Kokkos::DefaultExecutionSpace>
            _A("trsm_A", A.extent(0), A.extent(1) ) ; 
        Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{A.extent(0),A.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                _A(i,j) = A(i,j).val() ;                 
            }) ;
        if constexpr( rank_b == 1) {
            Kokkos::View<GRACE_REAL**, Kokkos::DefaultExecutionSpace>
                _B("trsm_B", B.extent(0), 1 ) ; 
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),1UL})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                _B(i,j) = B(i) ;                 
            }) ; 
            KokkosBlas::trsm(side,uplo,trans,diag,alpha,_A,_B) ; 
            // copy data back
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),B.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                B(i) = _B(i,j) ;                 
            }) ;
        } else {
            KokkosBlas::trsm(side,uplo,trans,diag,alpha,_A,B) ; 
        }
    } else if constexpr ( Sacado::IsFad<scalar_b_t>::value ) {
        GRACE_REAL const _alpha = alpha.val() ; 
        if constexpr( rank_b == 1) {
            Kokkos::View<GRACE_REAL**, Kokkos::DefaultExecutionSpace>
                _B("trsm_B", B.extent(0), 1 ) ; 
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),1UL})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                _B(i,j) = B(i).val() ;                 
            }) ; 
            KokkosBlas::trsm(side,uplo,trans,diag,_alpha,A,_B) ;
            // copy data back
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),B.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                B(i) = _B(i,j) ;                 
            }) ; 

        } else {
            Kokkos::View<GRACE_REAL**, Kokkos::DefaultExecutionSpace>
                _B("trsm_B", B.extent(0), B.extent(1) ) ;
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),B.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                _B(i,j) = B(i,j).val() ;                 
            }) ;
            KokkosBlas::trsm(side,uplo,trans,diag,_alpha,A,_B) ; 
            // copy data back
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),B.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                B(i,j) = _B(i,j) ;                 
            }) ;
        }
    } else {
        if constexpr( rank_b == 1) {
            Kokkos::View<GRACE_REAL**, Kokkos::DefaultExecutionSpace>
                _B("trsm_B", B.extent(0), 1 ) ; 
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),1UL})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                _B(i,j) = B(i) ;                 
            }) ; 
            KokkosBlas::trsm(side,uplo,trans,diag,alpha,A,_B) ; 
            // copy data back
            Kokkos::parallel_for("trsm_fill_matrices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0UL,0UL},{B.extent(0),B.extent(1)})
                            , KOKKOS_LAMBDA( int i, int j) 
            {
                B(i) = _B(i,j) ;                 
            }) ; 
        } else {
            KokkosBlas::trsm(side,uplo,trans,diag,alpha,A,B) ; 
        }
    }

}

}} /* namespace utils::linalg */

#endif 


