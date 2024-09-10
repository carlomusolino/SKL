/**
 * @file gmres.hh
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

#ifndef GRACE_ID_SOLVERS_GMRES_HH
#define GRACE_ID_SOLVERS_GMRES_HH

#include <grace_id_config.h>

#include <grace_id/utils/device.h>
#include <grace_id/utils/inline.h>
#include <grace_id/utils/types.hh>
#include <grace_id/utils/linalg.hh>
#include <grace_id/solvers/helpers.hh>

#include <Kokkos_Core.hpp>
#include<KokkosBlas1_team_nrm2.hpp>
#include<KokkosBlas1_nrm2.hpp>

#include <Sacado.hpp>

namespace grace {

class gmres {
    using thread_team_t = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> ; 
    using team_member_t = thread_team_t::member_type                        ; 

 public: 
    gmres( size_t problem_size, size_t max_iter, GRACE_REAL tol )
     : _N(problem_size), _max_iter(max_iter), _tol(tol) 
    {
        Kokkos::realloc(Q, _N, _max_iter, 2) ; 
        Kokkos::realloc(H, _max_iter+1, _max_iter ) ; 
        Kokkos::realloc(cs, _N) ; 
        Kokkos::realloc(sn, _N) ; 
    }

    template< typename res_t > 
    sfad_view_t<1> solve(res_t& res, sfad_view_t<1>& x ) 
    {
        using namespace Kokkos; 
        // Initialization: we batch the operations by launching 
        // a single thread team that will perform all operations. 
        thread_team_t policy(1,AUTO) ; 
        double b_norm, err; 
        
        auto q = subview(Q, ALL(), 0) ; 
        res.compute_residual(x, q) ;             // stored in-place: q = b 
        auto b_norm = utils::linalg::nrm2(q)     // norm of b
        auto Ax = res.compute_jvp(team, x, q) ;  // compute Ax  
        utils::linalg::axpy(-1, Ax, q) ;         // store in place: q = b - Ax 
        
        auto r_norm = utils::linalg::nrm2(q) ;   // Compute norm of q 

        double err = _r_norm / _b_norm ;         // Initial error 
        utils::linalg::scal(q, 1./r_norm, q) ;   // q = q/r_norm 

        int k { 0 } ; 
        do {
            // This call adds a column to H and Q 
            arnoldi_iteration(res, k) ; 
            // This call fills the rotation matrices
            givens_rotation(k) ; 

            parallel_for( "GMRES_update_beta", 1
                        , KOKKOS_LAMBDA(int dummy)
                {
                    beta(k+1) = -sn(k) * beta(k) ; 
                    beta(k ) *= cs(k) ; 
                    error     = Kokkos::fabs(beta(k+1)/b_norm) ; 
                }
            ) ; 
            if ( error < _tol ) {
                break ; 
            }
            k++ ; 
        } while( k < _max_iter) ; 

    } ; 

 private:

    template< typename res_t >
    void arnoldi_iteration(res_t&  res, sfad_view_t<1>& r, int n) {
        using namespace Kokkos ; 

        static constexpr double eps = 1e-12 ; 

        auto q = Kokkos::subview(Q, Kokkos::ALL(), n) ; 
        auto v = res.jvp( q)     ;  
        for(int j=0; j<=n; ++j) {
                auto q1  = Kokkos::subview(Q, Kokkos::ALL(), j) ; 
                H(j,n) = utils::linalg::dot(q1, v) ;  
                parallel_for( "Arnoldi_Gram_Schmidt_projection", _N 
                            , KOKKOS_LAMBDA (int l) 
                        {
                            v(l) = v(l) - H(j,n) * q1(l) ; 
                        }
                ) ; 
        }
        H(n+1,n) = utils::linalg::norm<2>(v) ;
        if ( H(n+1,n) > eps ) {
            parallel_for( "Arnoldi_normalize", _N 
                            , KOKKOS_LAMBDA (int l) 
                        {
                            Q(l,k) = v(l)/H(n+1,n) ; 
                        }
            ) ;
        } 
    }

    void givens_rotation(int n) {
        using namespace Kokkos ; 
        parallel_for( "GMRES_Givens_rotation", n-1
                    , KOKKOS_LAMBDA( int i ) 
            {
                GRACE_REAL tmp = cs(i) * H(i, n) + sn(i) * H(i+1, n) ; 
                H(i+1,n) = - sn(i) * H(i, n) + cs(i) * H(i+1, n)   ; 
                H(i,  n) = tmp ;     
            }
        ); 

        parallel_for( "GMRES_Givens_rotation", 1
                    , KOKKOS_LAMBDA( int _dummy ) 
            {
                GRACE_REAL v1 = H(n,  n) ; 
                GRACE_REAL v2 = H(n+1,n) ; 
                GRACE_REAL t = Kokkos::sqrt( math::int_pow<2>(v1) + math::int_pow<2>(v2) )
                cs(n) =  v1 / t ; 
                sn(n) =  v2 / t ; 
                H(n,n) = cs(n) + H(n,n) + sn(n) * H(n+1,n) ; 
                H(n+1,n) = 0. ; 
            }
        );
    }

    void compute_solution(int k) {

    }
         
    
    
    Kokkos::View<sfad_t<1>**, Kokkos::DeafultExecutionSpace> Q      ; 
    Kokkos::View<sfad_t<1>*, Kokkos::DeafultExecutionSpace>  cs, sn, beta ; 
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> H         ; //!< Hessenberg matrix

    size_t _N        ; //!< Size of the problem to invert 
    size_t _max_iter ; //!< Maximum number of iterations before restart
    GRACE_REAL _tol  ; //!< Absolute tolerance 
} ; 


}

#endif /* GRACE_ID_SOLVERS_GMRES_HH */