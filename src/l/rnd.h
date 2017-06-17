namespace l { namespace rnd { namespace d {
__device__ float mean0var1ii(float seed,  int u,  int v);
__device__ float mean0var1uu(float seed, uint u, uint v);

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
      
}}} /* l::rnd::d */
