/**
 * @file device.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-11
 * 
 * @copyright This file is part of SKL.
 * SKL is an evolution framework that uses Finite Difference 
 * methods to simulate relativistic astrophysical systems and plasma
 * dynamics.
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

#include <SKL_config.h>

#ifndef SKL_UTILS_DEVICE_H
#define SKL_UTILS_DEVICE_H

#if defined(SKL_ENABLE_CUDA) or defined (SKL_ENABLE_HIP)
#define SKL_DEVICE __device__ 
#define SKL_HOST   __host__ 
#define SKL_HOST_DEVICE __host__ __device__
#ifndef SKL_ALLOW_DEVICE_CONDITIONALS
#define DEVICE_CONDITIONAL(cond,a,b) ((static_cast<bool>(cond)) * a + (1-static_cast<bool>(cond)) * b) 
#else 
#define DEVICE_CONDITIONAL(cond,a,b) ((cond) ? a : b)
#endif
#else 
#define SKL_DEVICE 
#define SKL_HOST 
#define SKL_HOST_DEVICE 
#define DEVICE_CONDITIONAL(cond,a,b) ((cond) ? a : b)
#endif 


#endif 