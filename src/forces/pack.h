namespace forces { /* create Pa from all kinds of garbage */
inline __device__ void r3v3k2p(float x, float y, float z,
                               float vx, float vy, float vz,
                               int kind, Pa *p) {
    p->x = x;
    p->y = y;
    p->z = z;
    p->vx = vx;
    p->vy = vy;
    p->vz = vz;
    p->kind = kind;
}

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
inline __device__ void f2k2p(float2 d0, float2 d1, float2 d2, int kind, /**/ Pa *p) {
    p->x  = fst(d0);
    p->y  = scn(d0);
    p->z  = fst(d1);

    p->vx = scn(d1);
    p->vy = fst(d2);
    p->vz = scn(d2);

    p->kind = kind;
}

}
