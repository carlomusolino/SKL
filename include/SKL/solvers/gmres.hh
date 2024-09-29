/**
 * @file gmres.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-10
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * SKL is an evolution framework that uses Finite Volume
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

#ifndef SKL_SOLVERS_GMRES_HH
#define SKL_SOLVERS_GMRES_HH

#include <SKL_config.h>

#include <SKL/utils/device.h>
#include <SKL/utils/inline.h>
#include <SKL/utils/types.hh>
#include <SKL/utils/linalg.hh>
#include <SKL/solvers/helpers.hh>

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_team_nrm2.hpp>
#include <KokkosBlas1_nrm2.hpp>

#include <Sacado.hpp>

namespace skl {

class gmres {

 public: 
    gmres( size_t problem_size, size_t max_iter, SKL_REAL tol )
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
        /* Best accessed on host */
        auto h_H = create_mirror_view(H) ; 
        deep_copy(h_H, H) ; 

        auto q = Kokkos::subview(Q, Kokkos::ALL(), n) ; 
        auto v = res.jvp(q)     ;  
        for(int j=0; j<=n; ++j) {
                auto q1  = Kokkos::subview(Q, Kokkos::ALL(), j) ; 
                H(j,n) = utils::linalg::dot(q1, v) ;  
                utils::linalg::axpy(-H(j,n), q1, v) ; // Gram-Schmidt projection
                #if 0 
                parallel_for( "Arnoldi_Gram_Schmidt_projection", _N 
                            , KOKKOS_LAMBDA (int l) 
                        {
                            v(l) = v(l) - H(j,n) * q1(l) ; 
                        }
                ) ; 
                #endif 
        }
        H(n+1,n) = utils::linalg::nrm2(v) ;
        if ( H(n+1,n) > eps ) {
            auto qq = subview(Q, ALL(), k) ; 
            deep_copy(qq, v) ; 
            utils::linalg::scal(qq, 1./H(n+1,n)) ; // Rescale ( TODO scal in blas-1 interface )
        } 
    }

    void givens_rotation(int n) {
        using namespace Kokkos ; 
        parallel_for( "GMRES_Givens_rotation", n-1
                    , KOKKOS_LAMBDA( int i ) 
            {
                SKL_REAL tmp = cs(i) * H(i, n) + sn(i) * H(i+1, n) ; 
                H(i+1,n) = - sn(i) * H(i, n) + cs(i) * H(i+1, n)   ; 
                H(i,  n) = tmp ;     
            }
        ); 

        parallel_for( "GMRES_Givens_rotation", 1
                    , KOKKOS_LAMBDA( int _dummy ) 
            {
                SKL_REAL v1 = H(n,  n) ; 
                SKL_REAL v2 = H(n+1,n) ; 
                SKL_REAL t = Kokkos::sqrt( math::int_pow<2>(v1) + math::int_pow<2>(v2) )
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
    Kokkos::View<double**, Kokkos::HostExecutionSpace> H         ; //!< Hessenberg matrix ( stored on host )

    size_t _N        ; //!< Size of the problem to invert 
    size_t _max_iter ; //!< Maximum number of iterations before restart
    SKL_REAL _tol  ; //!< Absolute tolerance 
} ; 


}

#endif /* SKL_SOLVERS_GMRES_HH */