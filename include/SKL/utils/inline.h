/**
 * @file inline.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2023-03-22
 * 
 * @copyright This file is part of SKL.
 * SKL is an evolution framework that uses Finite Difference
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
#ifndef SKL_UTILS_INLINE_H
#define SKL_UTILS_INLINE_H

//**********************************************************************************************
//**********************************************************************************************
//**********************************************************************************************
/**
 * @brief Force inlining.
 * \ingroup utils
 */
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define SKL_FORCE_INLINE __forceinline
#else
#define SKL_FORCE_INLINE inline
#endif 
//**********************************************************************************************
//**********************************************************************************************
//**********************************************************************************************


//**********************************************************************************************
//**********************************************************************************************
//**********************************************************************************************
/**
 * @brief Direct compiler to inline.
 * \ingroup utils
 */
#if SKL_USE_ALWAYS_INLINE && defined(__GNUC__)
#define SKL_ALWAYS_INLINE inline __attribute__((always_inline))
#else 
#define SKL_ALWAYS_INLINE SKL_FORCE_INLINE 
#endif 
//**********************************************************************************************
//**********************************************************************************************
//**********************************************************************************************

#endif /* SKL_UTILS_INLINE_H */
