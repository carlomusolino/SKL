/**
 * @file coordinate_mapping.hpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-09
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

#ifndef GRACE_ID_MAPPINGS_COORDINATE_MAPPING_HH
#define GRACE_ID_MAPPINGS_COORDINATE_MAPPING_HH

#include <grace_id_config.h>

#include <grace_id/utils/device.h>
#include <grace_id/utils/inline.h>

namespace grace_id {



template< typename deriv_t >
class coordinate_mapping 
{
 public: 
    
    template< typename T >
    T GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() (T const& xp) {
        return static_cast<deriv_t const *> ( this ) ->template phys_to_log<T>(xp) ; 
    }

    template< typename T >
    T GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    inverse (T const& xp) {
        return static_cast<deriv_t const *> ( this ) ->template log_to_phys<T>(xp) ; 
    }

} ; 

}

#endif /* GRACE_ID_MAPPINGS_COORDINATE_MAPPING_HH */