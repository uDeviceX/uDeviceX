#ifndef VWALL_PAR_A
#error  VWALL_PAR_A is not set
#endif

#ifndef WVEL_PAR_W
#error  WVEL_PAR_W is not set
#endif

static void report0(float gd) {
    MSG("VWALL_SIN: gd = %6.2g", gd);
}

static void report(long i, long e, float gd) {
    bool cond;
    int freq;
    freq = WVEL_LOG_FREQ;
    cond = freq > 0 && i % freq == 0;
    if (cond) report0(gd);
}

static float gdot(float t) {
    float A, w;
    A = VWALL_PAR_A;
    w = WVEL_PAR_W;
    return A*sin(t);
}

float f0() {
    float t;
    t = 0;
    return gdot(t);
}

float  f(long i, long, float dt0) {
    float t;
    t = i*dt0;
    return gdot(t);
}
