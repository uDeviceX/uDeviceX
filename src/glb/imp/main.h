void sim()  {
    float gd;
    set_r0();

    gd = gdot::f0();
    set_gd(gd);
}
void step(long i, long e) {
    float gd;

    gd = gdot::f(i, e);
    set_gd(gd);
}
