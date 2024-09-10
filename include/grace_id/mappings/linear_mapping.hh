/**
 * @file linear_mapping.hh
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

#ifndef GRACE_ID_MAPPINGS_LINEAR_MAPPING_HH
#define GRACE_ID_MAPPINGS_LINEAR_MAPPING_HH

#include <grace_id_config.h>

#include <grace_id/mappings/coordinate_mapping.hh>

namespace grace_id {

class linear_coordinate_mapping 
 : public coordinate_mapping<linear_coordinate_mapping> 
{
 public:
    linear_coordinate_mapping( GRACE_REAL const _a, GRACE_REAL const _b )
     : a(_a), b(_b), ai(1./_a)
    {} 

    template < typename T>
    T GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    log_to_phys(T const& xl ) const {
        return (xl-b) * ai ; 
    }

    template < typename T>
    T GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    phys_to_log(T const& xp ) const {
        return a * xp + b ; 
    }   

 private: 
    GRACE_REAL a,b, ai ; 
} ; 

}

#endif 