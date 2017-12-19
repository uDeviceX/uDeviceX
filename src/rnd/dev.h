namespace rnd {
/* random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
   mean = 0, variance = 1 */

// passes of logistic map
static __const__ int N = 18;
// spacing coefficints for low discrepancy numbers
static __const__ float gold   = 0.6180339887498948482;
static __const__ float silver = 0.4142135623730950488;
static __const__ float bronze = 0.00008877875787352212838853023;
static __const__ float tin    = 0.00004602357186447026756768986;
static __const__ float sqrt2 = 1.41421356237309514547;

/************************* Trunk generator ***********************
 * Make one global random number per each timestep
 * cite G. Marsaglia
 * passes BigCrush
 *****************************************************************/

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

template<> __inline__ __device__ float __logistic_core<1>( float x ) {
    return FMA( 2.0 * x, x, -1.0 );
}

template<> __inline__ __device__ float __logistic_core<0>( float x ) {
    return x;
}

inline __device__ float mean0var1ii( float seed, int u, int v )
{
    float p = rem( ( ( u & 0x3FF ) * gold ) + u * bronze + ( ( v & 0x3FF ) * silver ) + v * tin ); // safe for large u or v
    float l = __logistic_core<N>( seed - p );
    return l * sqrt2;
}

inline __device__ float mean0var1uu( float seed, uint u, uint v )
{
    // 7 FLOPS
    float p = rem( ( ( u & 0x3FFU ) * gold ) + u * bronze + ( ( v & 0x3FFU ) * silver ) + v * tin ); // safe for large u or v
    // 45+1 FLOPS
    float l = __logistic_core<N>( seed - p );
    // 1 FLOP
    return l * sqrt2;
}

}
