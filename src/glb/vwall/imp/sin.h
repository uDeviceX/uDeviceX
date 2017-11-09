#ifndef WVEL_PAR_A
#error  WVEL_PAR_A is not set
#endif

#ifndef WVEL_PAR_W
#error  WVEL_PAR_W is not set
#endif

static float gdot(float t) {
    float A, w;
    A = WVEL_PAR_A;
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
