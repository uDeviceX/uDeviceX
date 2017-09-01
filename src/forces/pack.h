namespace forces { /* create Pa from all kind of garbage */
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

}
