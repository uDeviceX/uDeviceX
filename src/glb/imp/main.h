void sim()  {
    float gd;
    set_r0();

    gd = vwall::f0();
    set_gd(gd);
}
void step(long i, long e, float) {
    float gd;

    gd = vwall::f(i, e);
    set_gd(gd);
}
