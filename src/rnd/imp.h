namespace rnd {
// random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
// mean = 0
// variance = 1
// can be used directly for DPD
struct KISS {
    typedef uint32_t integer;
    integer x, y, z, c;

    KISS() : x( 0 ), y( 0 ), z( 0 ), c( 0 ) {}

    KISS( integer x_, integer y_, integer z_, integer c_ ) :
        x( x_ ), y( y_ ), z( z_ ), c( c_ ) {}

    float get_float();
private:    
    integer get_int();
};
}
