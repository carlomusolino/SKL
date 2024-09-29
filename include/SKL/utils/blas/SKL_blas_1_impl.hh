/**
 * @file linalg.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Some BLAS-1 routines for views of FadTypes.
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

#ifndef SKL_UTILS_BLAS_1_IMPL_HH
#define SKL_UTILS_BLAS_1_IMPL_HH

#include <SKL_config.h>

#include <SKL/utils/inline.h>
#include <SKL/utils/device.h>
#include <SKL/utils/types.hh>

#include <Kokkos_Core.hpp>
#include <Sacado.hpp> 

namespace utils { namespace linalg {

namespace impl {

template < typename T >
typename std::enable_if<std::is_scalar_v<T>, GRACE_REAL>::type 
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
scalarize(T const& x) 
{ return x ; }; 

template < typename T >
typename std::enable_if<Sacado::IsFad<T>::value, GRACE_REAL>::type 
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
scalarize(T const& x) 
{ return x.val() ; };


template< typename view_t >
GRACE_REAL GRACE_ALWAYS_INLINE 
_nrm2(view_t const & view )
{
    using scalar_t = typename view_t::non_const_value_type ; 
    GRACE_REAL res { 0. } ; 
    Kokkos::parallel_reduce("linalg::nrm2", view.extent(0)
                           , KOKKOS_LAMBDA (int i, GRACE_REAL& val)
            {
                val += scalarize<scalar_t>(view(i)) * scalarize<scalar_t>(view(i)) ; 
            }, Kokkos::Sum<GRACE_REAL>(res)) ; 
    return Kokkos::sqrt(res) ; 
}

template< typename team_t
        , typename view_t  >
GRACE_REAL GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
_nrm2(team_t team, view_t const & view )
{
    using scalar_t = typename view_t::non_const_value_type ; 
    GRACE_REAL res { 0. } ; 
    Kokkos::parallel_reduce("linalg::nrm2", Kokkos::TeamThreadRange(team, 0, view.extent(0))
                           , KOKKOS_LAMBDA (int i, GRACE_REAL& val)
            {
                val += scalarize<scalar_t>(view(i)) * scalarize<scalar_t>(view(i)) ; 
            }, Kokkos::Sum<GRACE_REAL>(res)) ; 
    return Kokkos::sqrt(res) ; 
}

template< typename view_a_t 
        , typename view_b_t >
GRACE_REAL GRACE_ALWAYS_INLINE 
_dot(view_a_t const & v,  view_b_t const & w)
{
    using scalar_a_t = typename view_a_t::non_const_value_type ; 
    using scalar_b_t = typename view_b_t::non_const_value_type ; 
    GRACE_REAL res { 0. } ; 
    Kokkos::parallel_reduce("linalg::dot", v.extent(0)
                           , KOKKOS_LAMBDA (int i, GRACE_REAL& val)
            {
                val += scalarize<scalar_a_t>(v(i)) * scalarize<scalar_b_t>(w(i)) ; 
            }, Kokkos::Sum<GRACE_REAL>(res)) ; 
    return res ; 
}

template< typename team_t 
        , typename view_a_t 
        , typename view_b_t >
GRACE_REAL GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
_dot(team_t team, view_a_t const & v,  view_b_t const & w)
{
    using scalar_a_t = typename view_a_t::non_const_value_type ; 
    using scalar_b_t = typename view_b_t::non_const_value_type ;
    GRACE_REAL res { 0. } ; 
    Kokkos::parallel_reduce("linalg::dot", Kokkos::TeamThreadRange(team, 0, v.extent(0))
                           , KOKKOS_LAMBDA (int i, GRACE_REAL& val)
            {
                val += scalarize<scalar_a_t>(v(i)) * scalarize<scalar_b_t>(w(i)) ; 
            }, Kokkos::Sum<GRACE_REAL>(res)) ; 
    return res ; 
}

template< typename out_view_t 
        , typename scalar_t 
        , typename in_view_t >
void  
_scal(out_view_t const& y, scalar_t const& alpha, in_view_t const& x)
{
    size_t constexpr rank_out = out_view_t::rank() ; 
    size_t constexpr rank_in  = in_view_t::rank()  ; 

    if constexpr ( rank_in == 1 ) {
        using out_scal_t = typename out_view_t::non_const_value_type ; 
        // ASSERT(x.extent(0) == y.extent(0)    ) ; 
        if constexpr ( Sacado::IsFad<out_scal_t>::value ) {
            Kokkos::parallel_for("linalg::scal", x.extent(0), 
            KOKKOS_LAMBDA(int i) 
            {
                y(i) = alpha * x(i) ; 
            }) ;
        } else {
            Kokkos::parallel_for("linalg::scal", x.extent(0), 
            KOKKOS_LAMBDA(int i) 
            {
                y(i) = scalarize(alpha) * scalarize(x(i)) ; 
            }) ;
        }
        
         
    } else {
        // ASSERT(x.extent(1) == alpha.extent(0)) ; 
        // ASSERT(x.extent(0) == y.extent(0)    ) ;
        using out_scal_t = typename out_view_t::non_const_value_type ; 

        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace> 
            policy( {0,0}, {x.extent(0), x.extent(1)} ) ;
        if constexpr ( Sacado::IsFad<out_scal_t>::value ) {
            Kokkos::parallel_for( "linalg::scal", policy, 
                KOKKOS_LAMBDA( int i, int j) 
            {
                y(i,j) = alpha(j) * x(i,j)  ;
            }) ; 
        } else {
            Kokkos::parallel_for( "linalg::scal", policy, 
                KOKKOS_LAMBDA( int i, int j) 
            {
                y(i,j) = scalarize(alpha(j)) * scalarize(x(i,j))  ;
            }) ;
        }
    }
    
}

template< typename out_view_t 
        , typename scalar_t 
        , typename in_view_t >
void  
_axpy(scalar_t const& alpha, in_view_t const& x, out_view_t const& y) 
{
    size_t constexpr rank_out = out_view_t::rank() ; 
    size_t constexpr rank_in  = in_view_t::rank()  ; 

    if constexpr ( rank_in == 1 ) {
        // ASSERT(x.extent(0) == y.extent(0)    ) ; 
        using out_scal_t = typename out_view_t::non_const_value_type ; 
        if constexpr ( Sacado::IsFad<out_scal_t>::value ) {
            Kokkos::parallel_for("linalg::axpy", x.extent(0), 
                    KOKKOS_LAMBDA(int i) 
            {
                y(i) += alpha * x(i) ; 
            }) ; 
        } else {
            Kokkos::parallel_for("linalg::axpy", x.extent(0), 
                    KOKKOS_LAMBDA(int i) 
            {
                y(i) += scalarize(alpha) * scalarize(x(i)) ; 
            }) ; 
        }
    } else {
        // ASSERT(x.extent(1) == alpha.extent(0)) ; 
        // ASSERT(x.extent(0) == y.extent(0)    ) ; 
        using out_scal_t = typename out_view_t::non_const_value_type ;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace> 
            policy( {0,0}, {x.extent(0), x.extent(1)} ) ; 
        if constexpr ( Sacado::IsFad<out_scal_t>::value ) {
            Kokkos::parallel_for( "linalg::axpy", policy, 
                KOKKOS_LAMBDA( int i, int j) 
            {
                y(i,j) += alpha(j) * x(i,j)  ;
            }) ; 
        } else {
            Kokkos::parallel_for( "linalg::axpy", policy, 
                KOKKOS_LAMBDA( int i, int j) 
            {
                y(i,j) += scalarize(alpha(j)) * scalarize(x(i,j))  ;
            }) ;
        }
    }
    
}


} /* namespace impl */




}}

#endif /* SKL_UTILS_BLAS_1_IMPL_HH */