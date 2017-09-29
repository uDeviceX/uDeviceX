static __device__ void pair(forces::Pa a, forces::Pa b, float rnd, /**/
                            float *fx, float *fy, float *fz) {
    forces::Fo f;
    f32f(fx, fy, fz, /**/ &f);
    forces::genf(a, b, rnd, /**/ f);
}
