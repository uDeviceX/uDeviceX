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
    m->z = (int)(y + ZS/2);
    m->dx = m->dy = m->dz = -1;
    m->i = -1;
}

enum {OK, END};
__device__ int nxt_xyz0(int *pdx, int *pdy, int *pdz) {
    int dx, dy, dz;
    dx = *pdx; dy = *pdy; dz = *pdz;
    if (++dx <= 1) goto ok; else dx = -1;
    if (++dy <= 1) goto ok; else dy = -1;
    if (++dz <= 1) goto ok; else goto end;
 ok:
    *pdx = dx; *pdy = dy; *pdz = dz;
    return OK;
 end:
    return END;
}

__device__ int valid(int dx0, int dy0, int dz0,
                     int dx , int dy, int dz) {
    return
        (dx0 == 0 || (dx0 == dx)) &&
        (dy0 == 0 || (dy0 == dy)) &&
        (dz0 == 0 || (dz0 == dz));
}

__device__ int d2i(int dx0, int dy0, int dz0,
                   int x, int y, int z) {
    x -= dx0*XS; y -= dy0*YS; z -= dz0*ZS;
    
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
