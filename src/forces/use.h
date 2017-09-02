namespace forces { /* set and get functions for Pa */
inline __device__ void p2r3(Pa *p, /**/ float *x, float *y, float *z) {
    *x = p->x;
    *y = p->y;
    *z = p->z;
}

inline __device__ void shift(float x, float y, float z, /**/ Pa *p) {
    p->x += x;
    p->y += y;
    p->z += z;
}

} /* namespace */
