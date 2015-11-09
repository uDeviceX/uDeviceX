//
//  logistic.h
//  cpudpd
//
//  Created by Dmitry Alexeev on 26/10/15.
//  Copyright © 2015 Dmitry Alexeev. All rights reserved.
//

#ifndef logistic_h
#define logistic_h

#include <immintrin.h>

#include "avxoperators.h"

namespace Logistic
{
	const __m256 veight  = _mm256_set1_ps(8.0f);
	const __m256 v_eight = _mm256_set1_ps(-8.0f);
	const __m256 vone    = _mm256_set1_ps(1.0f);
	const __m256 v_one   = _mm256_set1_ps(-1.0f);
	const __m256 vtwo    = _mm256_set1_ps(2.0f);
	
	inline __m256 rem( __m256 r ) {
		return r - _mm256_floor_ps( r );
	}
	
	inline  __m256 FMA( __m256 x, __m256 y, __m256 z )
	{
#ifdef __FMA__
		return _mm256_fmadd_ps(x, y, z);
#else
		return x * y + z;
#endif
	}
	
	template<int N> inline  __m256 __logistic_core( __m256 x )
	{
		__m256 x2 = x*x;
		__m256 r = FMA( FMA( veight, x2, v_eight ), x2, vone );
		return __logistic_core < N - 2 > ( r );
	}
	
	template<> inline  __m256 __logistic_core<1>( __m256 x ) {
		return FMA( vtwo * x, x, v_one );
	}
	
	template<> inline  __m256 __logistic_core<0>( __m256 x ) {
		return x;
	}
	
	// passes of logistic map
	const int N = 18;
	
	// spacing coefficints for low discrepancy numbers
	const float gold   = 0.6180339887498948482;
	const float silver = 0.4142135623730950488;
	const float sqrt2  = 1.41421356237309514547;
	
	const __m256 vgold   = _mm256_set1_ps(gold);
	const __m256 vsilver = _mm256_set1_ps(silver);
	const __m256 vsqrt2  = _mm256_set1_ps(sqrt2);
	
	inline __m256 mean0var1( __m256 seed, __m256 u, __m256 v )
	{
		__m256 p = rem( mysqrt(_mm256_min_ps(u, v)) * vgold + mysqrt(_mm256_max_ps(u, v)) * vsilver );
		__m256 l = __logistic_core<N>( seed - p );
		return l * vsqrt2;
	}
	
	inline __m256 mean0var1_unrolled( __m256 seed, __m256 u, __m256 v )
	{
		const __m256 p = rem( mysqrt(_mm256_min_ps(u, v)) * vgold + mysqrt(_mm256_max_ps(u, v)) * vsilver );
		__m256 x = seed - p;
		
		__m256 x2;
		x2 = x*x;
		x = FMA( FMA( veight, x2, v_eight ), x2, vone );
		x2 = x*x;
		x = FMA( FMA( veight, x2, v_eight ), x2, vone );
		x2 = x*x;
		x = FMA( FMA( veight, x2, v_eight ), x2, vone );
		x2 = x*x;
		x = FMA( FMA( veight, x2, v_eight ), x2, vone );
		x2 = x*x;
		x = FMA( FMA( veight, x2, v_eight ), x2, vone );
		x2 = x*x;
		x = FMA( FMA( veight, x2, v_eight ), x2, vone );
		x2 = x*x;
		x = FMA( FMA( veight, x2, v_eight ), x2, vone );
		x2 = x*x;
		x = FMA( FMA( veight, x2, v_eight ), x2, vone );
		x2 = x*x;
		x = FMA( FMA( veight, x2, v_eight ), x2, vone );
		
		return x * vsqrt2;
	}
	
#define LOGCORE(_ID)\
{xx##_ID = _mm256_mul_ps(x##_ID, x##_ID); \
x##_ID = FMA( FMA( veight, xx##_ID, v_eight ), xx##_ID, vone );}
	
#define ASSIGN(_ID)\
{r##_ID = x##_ID * vsqrt2;}
	
	__attribute__((always_inline)) static void mean0var1_unrolledx4(__m256 &r1, __m256 &r2, __m256 &r3, __m256 &r4,
																	__m256 seed, __m256 u, __m256 v1, __m256 v2, __m256 v3, __m256 v4)
	{
		const __m256 p1 = rem( mysqrt(_mm256_min_ps(u, v1)) * vgold + mysqrt(_mm256_max_ps(u, v1)) * vsilver );
		const __m256 p2 = rem( mysqrt(_mm256_min_ps(u, v2)) * vgold + mysqrt(_mm256_max_ps(u, v2)) * vsilver );
		const __m256 p3 = rem( mysqrt(_mm256_min_ps(u, v3)) * vgold + mysqrt(_mm256_max_ps(u, v3)) * vsilver );
		const __m256 p4 = rem( mysqrt(_mm256_min_ps(u, v4)) * vgold + mysqrt(_mm256_max_ps(u, v4)) * vsilver );
				
		__m256 xx1, xx2, xx3, xx4;
		__m256 x1 = seed - p1;
		__m256 x2 = seed - p2;
		__m256 x3 = seed - p3;
		__m256 x4 = seed - p4;

		LOGCORE(1);
		LOGCORE(2);
		LOGCORE(3);
		LOGCORE(4);
		
		LOGCORE(1);
		LOGCORE(2);
		LOGCORE(3);
		LOGCORE(4);
		
		LOGCORE(1);
		LOGCORE(2);
		LOGCORE(3);
		LOGCORE(4);
		
		LOGCORE(1);
		LOGCORE(2);
		LOGCORE(3);
		LOGCORE(4);
		
		LOGCORE(1);
		LOGCORE(2);
		LOGCORE(3);
		LOGCORE(4);
		
		LOGCORE(1);
		LOGCORE(2);
		LOGCORE(3);
		LOGCORE(4);
		
		LOGCORE(1);
		LOGCORE(2);
		LOGCORE(3);
		LOGCORE(4);
		
		LOGCORE(1);
		LOGCORE(2);
		LOGCORE(3);
		LOGCORE(4);
		
		LOGCORE(1);
		LOGCORE(2);
		LOGCORE(3);
		LOGCORE(4);
			
		ASSIGN(1);
		ASSIGN(2);
		ASSIGN(3);
		ASSIGN(4);
	}

}

namespace LogisticNonvec
{
	inline float rem( float r ) {
		return r - floorf( r );
	}
	
	// FMA wrapper for the convenience of switching rouding modes
	inline float FMA( float x, float y, float z )
	{
		return x*y + z;
	}
	
	template<int N> inline  float __logistic_core( float x )
	{
		float x2 = x * x;
		float r = FMA( FMA( 8.0f, x2, -8.0f ), x2, 1.0f );
		return __logistic_core < N - 2 > ( r );
	}
	
	template<> inline  float __logistic_core<1>( float x ) {
		return FMA( 2.0f * x, x, -1.0f );
	}
	
	template<> inline  float __logistic_core<0>( float x ) {
		return x;
	}
	
	// passes of logistic map
	const static int N = 18;
	
	// spacing coefficints for low discrepancy numbers
	const static float gold   = 0.6180339887498948482;
	const static float silver = 0.4142135623730950488;
	const static float sqrt2  = 1.41421356237309514547;
	
	inline float mean0var1( float seed, float u, float v )
	{
		float tmp[8];
		float su, sv;
		__m256 vu =_mm256_set1_ps(u);
		_mm256_store_ps(tmp, mysqrt(vu));
		su = tmp[0];
		
		__m256 vv =_mm256_set1_ps(v);
		_mm256_store_ps(tmp, mysqrt(vv));
		sv = tmp[0];
		
		float p = rem( (std::min(su,sv)) * gold + (std::max(su,sv)) * silver );
		
		//float p = rem( sqrt(std::min(u,v)) * gold + sqrt(std::max(u,v)) * silver );
		float l = __logistic_core<N>( seed - p );
		return l * sqrt2;
	}
	
	inline float mean0var1_orig( float seed, float u, float v )
	{
		float p = rem( sqrt(std::min(u,v)) * gold + sqrt(std::max(u,v)) * silver );
		float l = __logistic_core<N>( seed - p );
		return l * sqrt2;
	}
}



#endif /* logistic_h */
