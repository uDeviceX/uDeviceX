namespace map {

struct M {
    const hforces::Frag *frag;
    int dx, dy, dz;
    int x, y, z; /* center */
    int i;
    int end;
};

__device__ void ini(hforces::Frag *frag, float x, float y, float z, /**/ M *m) {
    m->frag = frag;
    m->x = (int)(x + XS/2);
    m->y = (int)(y + YS/2);
    m->z = (int)(y + YS/2);
    m->dx = m->dy = m->dz = -1;
    m->i = -1;
}

__device__ int nxt(/**/ M* m) {
    if (m->i == -1) {
        return 0;
    } else {
        return 1;
    }
}

__device__ int endp(int i, M *m) {
    return 0;
}

} /* namespace */
