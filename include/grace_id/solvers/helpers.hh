/**
 * @file helpers.hh
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

#ifndef GRACE_ID_SOLVERS_HELPERS_HH
#define GRACE_ID_SOLVERS_HELPERS_HH

#include <grace_id_config.h>

#include <grace_id/utils/device.h>
#include <grace_id/utils/inline.h>

#include <Kokkos_Core.hpp>
#include <Sacado.hpp>

namespace utils {

GRACE_REAL GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
norm( grace::fad_view<GRACE_REAL> v ) {
    GRACE_REAL sum{0} ; 
    for( int ii=0; ii<v.extent(0); ++vv) {
        sum += math::int_pow<2>(v(ii).val()) ; 
    }
    return Kokkos::sqrt(sum) ; 
}

}

#endif 