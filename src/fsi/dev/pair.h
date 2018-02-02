template <typename Par>
static __device__ void pair0(Par params, const Pa a, const Pa b, float rnd, /**/ float *fx, float *fy, float *fz) {
    PairFo f;
    pair_force(params, a, b, rnd, /**/ &f);
    *fx = f.x; *fy = f.y; *fz = f.z;
}

template <typename Par>
static __device__ void pair(Par params, const Pa a, const Pa b, float rnd, /**/
                            float *fx, float *fy, float *fz,
                            Fo f) {
    /* f[xyz]: local force; Fo f: remote force */
    float x, y, z; /* pair force */
    pair0(params, a, b, rnd, /**/ &x, &y, &z);
    *fx += x; *fy += y; *fz += z;
    atomicAdd(f.x, -x); atomicAdd(f.y, -y); atomicAdd(f.z, -z);
}
