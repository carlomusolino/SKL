/**
 * @file linalg.hh
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
 * 
 * 
 */

#ifndef GRACE_ID_UTILS_BLAS_1_HH
#define GRACE_ID_UTILS_BLAS_1_HH

#include <grace_id_config.h>

#include <grace_id/utils/inline.h>
#include <grace_id/utils/types.hh>
#include <grace_id/utils/blas/grace_blas_1_impl.hh>

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_nrm2.hpp> 
#include <KokkosBlas1_team_nrm2.hpp> 
#include <KokkosBlas1_dot.hpp> 
#include <KokkosBlas1_team_dot.hpp> 
#include <KokkosBlas1_scal.hpp> 
#include <KokkosBlas1_axpby.hpp> 

#include <Sacado.hpp>

namespace utils { namespace linalg {

/**
 * @brief Compute 2-norm of a vector.
 * 
 * \ingroup blas
 * 
 * This function will call the KokkosBlas implementation 
 * if the underlying type allows to do so, otherwise it
 * will call a custom implementation.
 * 
 * @tparam view_t Type of View representing the vector. 
 * @param view    View representing the vector.
 * @return GRACE_REAL The 2-norm of the input vector.
 */
template< typename view_t >
GRACE_REAL GRACE_ALWAYS_INLINE 
nrm2(view_t const & view )
{
    static_assert( Kokkos::is_view<view_t>::value, "view_t must be a Kokkos::View.");
    using scalar_t = typename view_t::non_const_value_type ; 
    if constexpr( Sacado::IsFad<scalar_t>::value ) {
        return impl::_nrm2(view) ; 
    } else {
        return KokkosBlas::nrm2(view) ; 
    }
}

/**
 * @brief Compute 2-norm of a vector within a ThreadTeam parallel environment.
 * 
 * \ingroup blas
 * 
 * This function will call the KokkosBlas implementation 
 * if the underlying type allows to do so, otherwise it
 * will call a custom implementation.
 * 
 * @tparam team_t Type of ThreadTeam member.
 * @tparam view_t Type of View representing the vector. 
 * @param team    Thread team.
 * @param view    View representing the vector.
 * @return GRACE_REAL The 2-norm of the input vector.
 */
template< typename team_t
        , typename view_t  >
GRACE_REAL GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
nrm2(team_t team, view_t const & view ) {
    static_assert( Kokkos::is_view<view_t>::value, "view_t must be a Kokkos::View.");
    using scalar_t = typename view_t::non_const_value_type ; 
    if constexpr( Sacado::IsFad<scalar_t>::value ) {
        return impl::_nrm2(team,view) ; 
    } else {
        return KokkosBlas::Experimental::nrm2(team, view) ; 
    }
}

/**
 * @brief Compute dot product of two vectors.
 * \ingroup blas
 * 
 * This function will call the KokkosBlas implementation 
 * if the underlying type allows to do so, otherwise it
 * will call a custom implementation.
 * @tparam view_a_t Type of View representing vector A.
 * @tparam view_b_t Type of View representing vector B. 
 * @param v Vector A.
 * @param w Vector B.
 * @return GRACE_REAL The dot product of the two vectors.
 */
template< typename view_a_t 
        , typename view_b_t >
GRACE_REAL GRACE_ALWAYS_INLINE 
dot(view_a_t const & v,  view_b_t const & w) {
    static_assert( Kokkos::is_view<view_a_t>::value, "view_a_t must be a Kokkos::View.");
    static_assert( Kokkos::is_view<view_b_t>::value, "view_b_t must be a Kokkos::View.");
    using scalar_a_t = typename view_a_t::non_const_value_type ; 
    using scalar_b_t = typename view_b_t::non_const_value_type ; 

    if constexpr ( Sacado::IsFad<scalar_a_t>::value or Sacado::IsFad<scalar_b_t>::value ) {
        return impl::_dot(v,w) ; 
    } else {
        return KokkosBlas::dot(v,w) ; 
    }
}

/**
 * @brief Compute dot product of two vectors in a ThreadTeam parallel environment.
 * \ingroup blas
 * 
 * This function will call the KokkosBlas implementation 
 * if the underlying type allows to do so, otherwise it
 * will call a custom implementation.
 * 
 * @tparam team_t   Type of thread team.
 * @tparam view_a_t Type of View representing vector A.
 * @tparam view_b_t Type of View representing vector B. 
 * @param team Thread team member.
 * @param v Vector A.
 * @param w Vector B.
 * @return GRACE_REAL The dot product of the two vectors.
 */
template< typename team_t 
        , typename view_a_t 
        , typename view_b_t >
GRACE_REAL GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
dot(team_t team, view_a_t const & v,  view_b_t const & w) {
    static_assert( Kokkos::is_view<view_a_t>::value, "view_a_t must be a Kokkos::View.");
    static_assert( Kokkos::is_view<view_b_t>::value, "view_b_t must be a Kokkos::View.");
    using scalar_a_t = typename view_a_t::non_const_value_type ; 
    using scalar_b_t = typename view_b_t::non_const_value_type ; 

    if constexpr ( Sacado::IsFad<scalar_a_t>::value or Sacado::IsFad<scalar_b_t>::value ) {
        return impl::_dot(team,v,w) ; 
    } else {
        return KokkosBlas::dot(team,v,w) ; 
    }
}

template< typename out_view_t 
        , typename scalar_t 
        , typename in_view_t >
void GRACE_ALWAYS_INLINE
scal(out_view_t const& y, scalar_t const& alpha, in_view_t const& x) 
{
    /* Let's do some checks on the inputs! */
    static_assert( Kokkos::is_view<out_view_t>::value, "out_view_t must be a Kokkos::View.");
    static_assert( Kokkos::is_view<in_view_t>::value, "in_view_t must be a Kokkos::View.");

    using scalar_out_t = typename out_view_t::non_const_value_type ; 
    using scalar_in_t  = typename in_view_t::non_const_value_type  ; 

    size_t constexpr rank_out = out_view_t::rank() ; 
    size_t constexpr rank_in  = in_view_t::rank()  ; 
    static_assert( rank_in == rank_out, "In scal, output has different rank from input.") ; 
    static_assert( rank_in == 1 or rank_in == 2, "In scal, ranks are different from 1 or 2.") ;
    
    if constexpr ( rank_in == 1 ) {
        // If the views are rank 1 then alpha is a true scalar 
        using non_const_scalar_t = typename std::remove_cvref_t<scalar_t > ; 
        if constexpr (   Sacado::IsFad<scalar_out_t>::value
                    or   Sacado::IsFad<scalar_in_t>::value
                    or   Sacado::IsFad<non_const_scalar_t>::value ) { 
            impl::_scal(y,alpha,x) ; 
        } else {
            KokkosBlas::scal(y,alpha,x) ; 
        }  
    } else {
        // otherwise alpha is a rank 1 view itself 
        static_assert(Kokkos::is_view<scalar_t>::value, "If x and y are rank 2 then alpha must be a Kokkos::View.") ; 
        static_assert(scalar_t::rank() == 1, "If x and y are rank 2 then alpha must be rank 1.") ; 
        using non_const_scalar_t = typename scalar_t::non_const_value_type ; 
        if constexpr (   Sacado::IsFad<scalar_out_t>::value
                    or   Sacado::IsFad<scalar_in_t>::value
                    or   Sacado::IsFad<non_const_scalar_t>::value ) { 
            impl::_scal(y,alpha,x) ; 
        } else {
            KokkosBlas::scal(y,alpha,x) ; 
        }  
    }
    
}

template< typename out_view_t 
        , typename scalar_t 
        , typename in_view_t >
void GRACE_ALWAYS_INLINE
axpy(scalar_t const& alpha, in_view_t const& x, out_view_t const& y) 
{
    /* Let's do some checks on the inputs! */
    static_assert( Kokkos::is_view<out_view_t>::value, "out_view_t must be a Kokkos::View.");
    static_assert( Kokkos::is_view<in_view_t>::value, "in_view_t must be a Kokkos::View.");

    using scalar_out_t = typename out_view_t::non_const_value_type ; 
    using scalar_in_t  = typename in_view_t::non_const_value_type  ; 

    size_t constexpr rank_out = out_view_t::rank() ; 
    size_t constexpr rank_in  = in_view_t::rank()  ;

    static_assert( rank_in == rank_out, "In scal, output has different rank from input.") ; 
    static_assert( rank_in == 1 or rank_in == 2, "In scal, ranks are different from 1 or 2.") ;

    if constexpr ( rank_in == 1 ) {
        // If the views are rank 1 then alpha is a true scalar 
        using non_const_scalar_t = typename std::remove_cvref_t<scalar_t > ; 
        if constexpr (   Sacado::IsFad<scalar_out_t>::value
                    or   Sacado::IsFad<scalar_in_t>::value
                    or   Sacado::IsFad<non_const_scalar_t>::value ) { 
            impl::_axpy(alpha,x,y) ; 
        } else {
            KokkosBlas::axpy(alpha,x,y) ; 
        }  
    } else {
        // otherwise alpha is a rank 1 view itself 
        static_assert(Kokkos::is_view<scalar_t>::value, "If x and y are rank 2 then alpha must be a Kokkos::View.") ; 
        static_assert(scalar_t::rank() == 1, "If x and y are rank 2 then alpha must be rank 1.") ; 
        using non_const_scalar_t = typename scalar_t::non_const_value_type ; 
        if constexpr (   Sacado::IsFad<scalar_out_t>::value
                    or   Sacado::IsFad<scalar_in_t>::value
                    or   Sacado::IsFad<non_const_scalar_t>::value ) { 
            impl::_axpy(alpha,x,y) ; 
        } else {
            KokkosBlas::axpy(alpha,x,y) ; 
        }  
    }
    
}

}} 

#endif /* GRACE_ID_UTILS_LINALG_HH */