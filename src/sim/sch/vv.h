static __device__ void update0(float m, const float *f, /**/ float *r, float *v) {
    v[0] += f[0]/m*dt;
    v[1] += f[1]/m*dt;
    v[2] += f[2]/m*dt;

    r[0] += v[0]*dt;
    r[1] += v[1]*dt;
    r[2] += v[2]*dt;
}
