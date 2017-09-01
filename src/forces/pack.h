namespace forces { /* create Pa from all kinds of garbage */
inline __device__ void rvk2p(float r[3], float v[3], int kind, /**/ Pa *p) {
    enum {X, Y, Z};
    p->x = r[X];
    p->y = r[Y];
    p->z = r[Z];

    p->vx = v[X];
    p->vy = v[Y];
    p->vz = v[Z];

    p->kind = kind;
}

inline __device__ float fst(float2 t) { return t.x; }
inline __device__ float scn(float2 t) { return t.y; }
inline __device__ void f2k2p(float2 d0, float2 d1, int kind, /**/ Pa *p) {
}

}
