namespace rnd {
// random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
// mean = 0
// variance = 1
// can be used directly for DPD
struct KISS {
    typedef uint32_t integer;
    integer x, y, z, c;
    KISS(integer x_, integer y_, integer z_, integer c_);
    float get_float();
private:    
    integer get_int();
    KISS();
};
}
