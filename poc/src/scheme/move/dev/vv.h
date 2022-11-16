static __device__ void update0(float dt, float m, const float *f, /**/ float *r, float *v) {
    enum {X, Y, Z};
    v[X] += f[X]/m*dt;
    v[Y] += f[Y]/m*dt;
    v[Z] += f[Z]/m*dt;

    r[X] += v[X]*dt;
    r[Y] += v[Y]*dt;
    r[Z] += v[Z]*dt;
}
