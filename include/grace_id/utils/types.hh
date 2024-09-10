/**
 * @file types.hh
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

#ifndef GRACE_ID_UTILS_TYPES_HH
#define GRACE_ID_UTILS_TYPES_HH

#include <grace_id_config.h>

#include <Sacado.hpp>
#include <Kokkos_Core.hpp>

namespace grace {

template< size_t n_der >
using sfad_t = Sacado::Fad::SFad<GRACE_REAL,n_der> ; 

using fad_t  = Sacado::Fad::DFad<GRACE_REAL>        ; 

template < size_t n_der >
using sfad_view_t = Kokkos::View<sfad_t<n_der>*, Kokkos::DefaultExecutionSpace> ;

}

#endif 