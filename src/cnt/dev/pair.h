static __device__ void pair(forces::Pa a, forces::Pa b, float rnd, /**/
                            float *fx, float *fy, float *fz) {
    forces::Fo f;
    forces::gen(a, b, rnd, /**/ &f);
    *fx = f.x; *fy = f.y; *fz = f.z;
}
