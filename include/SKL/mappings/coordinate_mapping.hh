/**
 * @file coordinate_mapping.hpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-09
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

#ifndef SKL_MAPPINGS_COORDINATE_MAPPING_HH
#define SKL_MAPPINGS_COORDINATE_MAPPING_HH

#include <SKL_config.h>

#include <SKL/utils/device.h>
#include <SKL/utils/inline.h>

namespace skl {



template< typename deriv_t >
class coordinate_mapping 
{
 public: 
    
    template< typename T >
    T SKL_ALWAYS_INLINE SKL_HOST_DEVICE 
    operator() (T const& xp) {
        return static_cast<deriv_t const *> ( this ) ->template phys_to_log<T>(xp) ; 
    }

    template< typename T >
    T SKL_ALWAYS_INLINE SKL_HOST_DEVICE 
    inverse (T const& xp) {
        return static_cast<deriv_t const *> ( this ) ->template log_to_phys<T>(xp) ; 
    }

} ; 

}

#endif /* SKL_MAPPINGS_COORDINATE_MAPPING_HH */