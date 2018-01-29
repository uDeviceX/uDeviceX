static __device__ void update0(MoveParams_v parv, float m, const float *f, /**/ float *r, float *v) {
    enum {X, Y, Z};

    float dt0 = parv.dt0;
    assert(dt0>=0.95*dt && dt0<=1.05*dt);

    r[X] += v[X]*dt0;
    r[Y] += v[Y]*dt0;
    r[Z] += v[Z]*dt0;

    v[X] += f[X]/m*dt0;
    v[Y] += f[Y]/m*dt0;
    v[Z] += f[Z]/m*dt0;
}
