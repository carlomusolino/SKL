/**
 * @file helpers.hh
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

#ifndef SKL_SOLVERS_HELPERS_HH
#define SKL_SOLVERS_HELPERS_HH

#include <SKL_config.h>

#include <SKL/utils/device.h>
#include <SKL/utils/inline.h>

#include <Kokkos_Core.hpp>
#include <Sacado.hpp>

namespace utils {

SKL_REAL SKL_ALWAYS_INLINE SKL_HOST_DEVICE 
norm( skl::fad_view<SKL_REAL> v ) {
    SKL_REAL sum{0} ; 
    for( int ii=0; ii<v.extent(0); ++vv) {
        sum += math::int_pow<2>(v(ii).val()) ; 
    }
    return Kokkos::sqrt(sum) ; 
}

}

#endif 