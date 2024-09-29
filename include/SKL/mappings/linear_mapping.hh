/**
 * @file linear_mapping.hh
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

#ifndef SKL_MAPPINGS_LINEAR_MAPPING_HH
#define SKL_MAPPINGS_LINEAR_MAPPING_HH

#include <SKL_config.h>

#include <SKL/mappings/coordinate_mapping.hh>

namespace skl {

class linear_coordinate_mapping 
 : public coordinate_mapping<linear_coordinate_mapping> 
{
 public:
    linear_coordinate_mapping( SKL_REAL const _a, SKL_REAL const _b )
     : a(_a), b(_b), ai(1./_a)
    {} 

    template < typename T>
    T SKL_ALWAYS_INLINE SKL_HOST_DEVICE 
    log_to_phys(T const& xl ) const {
        return (xl-b) * ai ; 
    }

    template < typename T>
    T SKL_ALWAYS_INLINE SKL_HOST_DEVICE 
    phys_to_log(T const& xp ) const {
        return a * xp + b ; 
    }   

 private: 
    SKL_REAL a,b, ai ; 
} ; 

}

#endif 