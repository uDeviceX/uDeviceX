void sim()  {
    float gd;
    set_r0();

    gd = gdot::f0();
    set_gd(gd);
}
void step(long s, long e) {
    float gd;

    gd = gdot::f(s, e);
    set_gd(gd);
}
