static __device__ void update0(float dt0, MoveParams_v parv, float m, const float *f, /**/ float *r, float *v) {
    enum {X, Y, Z};
    v[X] += f[X]/m*dt0;
    v[Y] += f[Y]/m*dt0;
    v[Z] += f[Z]/m*dt0;

    r[X] += v[X]*dt0;
    r[Y] += v[Y]*dt0;
    r[Z] += v[Z]*dt0;
}
