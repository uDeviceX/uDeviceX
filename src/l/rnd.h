#include <limits>
#include <stdint.h>
#include "dpd/tiny-float.h"

namespace l { namespace rnd { namespace d {
// random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
// mean = 0
// variance = 1
// can be used directly for DPD

// passes of logistic map
const static int N = 18;
// spacing coefficints for low discrepancy numbers
const static float gold   = 0.6180339887498948482;
const static float silver = 0.4142135623730950488;
const static float bronze = 0.00008877875787352212838853023;
const static float tin    = 0.00004602357186447026756768986;
const static float sqrt2 = 1.41421356237309514547;

/************************* Trunk generator ***********************
 * Make one global random number per each timestep
 * cite G. Marsaglia
 * passes BigCrush
 *****************************************************************/
struct KISS {
  typedef uint32_t integer;
  integer x, y, z, c;

  KISS() : x( 0 ), y( 0 ), z( 0 ), c( 0 ) {}

  KISS( integer x_, integer y_, integer z_, integer c_ ) :
      x( x_ ), y( y_ ), z( z_ ), c( c_ ) {}

  float get_float()
  {
    return get_int() / float( std::numeric_limits<integer>::max() );
  }

  integer get_int()
  {
    uint64_t t, a = 698769069ULL;
    x = 69069 * x + 12345;
    y ^= ( y << 13 );
    y ^= ( y >> 17 );
    y ^= ( y << 5 ); /* y must never be set to zero! */
    t = a * z + c;
    c = ( t >> 32 ); /* Also avoid setting z=c=0! */
    return x + y + ( z = t );
  }
};

#ifdef __CUDACC__

/************************* Branch generator **********************
 * Make one random number per pair of particles per timestep
 * Based on the Logistic map on interval [-1,1]
 * each branch no weaker than trunk
 * zero dependency between branches
 *****************************************************************/

// floating point version of LCG
__inline__ __device__ float rem( float r ) {
  return r - floorf( r );
}

// FMA wrapper for the convenience of switching rouding modes
__inline__ __device__ float FMA( float x, float y, float z ) {
  return __fmaf_rz( x, y, z );
}

// logistic rounds
// <3> : 4 FMA + 1 MUL
// <2> : 2 FMA + 1 MUL
// <1> : 1 FMA + 1 MUL

template<int N> __inline__ __device__ float __logistic_core( float x )
{
  float x2 = x * x;
  float r = FMA( FMA( 8.0, x2, -8.0 ), x2, 1.0 );
  return __logistic_core < N - 2 > ( r );
}
template<int N> struct __logistic_core_flops_counter {
const static unsigned long long FLOPS = 5 + __logistic_core_flops_counter<N-2>::FLOPS;
};

template<> __inline__ __device__ float __logistic_core<1>( float x ) {
return FMA( 2.0 * x, x, -1.0 );
}
template<> struct __logistic_core_flops_counter<1> {
const static unsigned long long FLOPS = 3;
};

template<> __inline__ __device__ float __logistic_core<0>( float x ) {
return x;
}
template<> struct __logistic_core_flops_counter<0> {
const static unsigned long long FLOPS = 0;
};

__inline__ __device__ float mean0var1ii( float seed, int u, int v )
{
  float p = rem( ( ( u & 0x3FF ) * gold ) + u * bronze + ( ( v & 0x3FF ) * silver ) + v * tin ); // safe for large u or v
  float l = __logistic_core<N>( seed - p );
  return l * sqrt2;
}

__inline__ __device__ float mean0var1uu( float seed, uint u, uint v )
{
  // 7 FLOPS
  float p = rem( ( ( u & 0x3FFU ) * gold ) + u * bronze + ( ( v & 0x3FFU ) * silver ) + v * tin ); // safe for large u or v
  // 45+1 FLOPS
  float l = __logistic_core<N>( seed - p );
  // 1 FLOP
  return l * sqrt2;
}

__inline__ __device__ float mean0var1_dual( float seed, float u, float v )
{
  float p = rem( sqrtf(u) * gold + sqrtf(v) * silver ); // Acknowledging Dmitry for the use of sqrtf
  float l = __logistic_core<N>( seed - p );
  float z = __logistic_core<N>( seed + p - 1.f );
  return l + z;
}

#endif
}}} /* l:rnd:d */
