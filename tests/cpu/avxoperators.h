//
//  avxoperators.h
//  cpudpd
//
//  Created by Dmitry Alexeev on 30/10/15.
//  Copyright © 2015 Dmitry Alexeev. All rights reserved.
//

#ifndef avxoperators_h
#define avxoperators_h

#include <immintrin.h>


#ifdef __INTEL_COMPILER

inline __m256 operator + (__m256 a, __m256 b)
{
	return _mm256_add_ps(a, b);
}

inline __m256 operator * (__m256 a, __m256 b)
{
	return _mm256_mul_ps(a, b);
}

inline __m256 operator - (__m256 a, __m256 b)
{
	return _mm256_sub_ps(a, b);
}

inline __m256 operator / (__m256 a, __m256 b)
{
	return _mm256_div_ps(a, b);
}

inline __m256 operator - (__m256 a)
{
	return _mm256_sub_ps(_mm256_setzero_ps(), a);
}

inline __m256 operator += (__m256 &a, __m256 b)
{
	a = _mm256_add_ps(a, b);
	return a;
}

#endif

inline __m256 mysqrt(__m256 a)
{
	// One Newton-Raphson iteration
	
	const __m256 vhalf = _mm256_set1_ps(0.5f);
	
	const __m256 invx0 = _mm256_rsqrt_ps(a);
	const __m256 x0    = _mm256_rcp_ps(invx0);
	
#ifdef __FMA__
	const __m256 tmp = _mm256_fmadd_ps(a, invx0, x0);
#else
	const __m256 tmp = a * invx0 + x0;
#endif
	
	return _mm256_set1_ps(0.5f) * tmp;
}

inline __m256 myrsqrt(__m256 a)
{
	// One Newton-Raphson iteration
	
	const __m256 x0       = _mm256_rsqrt_ps(a);
	const __m256 half_a_x = _mm256_set1_ps(0.5f) * a * x0;
	
#ifdef __FMA__
	const __m256 tmp = _mm256_fnmadd_ps(half_a_x, x0, _mm256_set1_ps(1.5f));
#else
	const __m256 tmp =  _mm256_set1_ps(1.5f) - half_a_x * x0;
#endif
	
	return x0 * tmp;
}



#endif /* avxoperators_h */
