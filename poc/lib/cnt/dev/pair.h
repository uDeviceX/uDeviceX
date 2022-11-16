template <typename Par>
static __device__ void pair(Par params, PairPa a, PairPa b, float rnd, /**/
                            float *fx, float *fy, float *fz) {
    PairFo f;
    pair_force(&params, a, b, rnd, /**/ &f);
    *fx = f.x; *fy = f.y; *fz = f.z;
}
