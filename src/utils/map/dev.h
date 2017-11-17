namespace map {

enum {OK, END, OLD, NEW};
struct M {
    int dx0, dy0, dz0;
    const int *start;
    int dx, dy, dz;
    int x, y, z; /* center */
    int i;
    int end;
};

__device__ void ini0(hforces::Frag *frag,
                     float x, float y, float z,
                     int xs, int ys, int zs,
                     /**/ M *m) {
    m->x = (int)(x + xs/2);
    m->y = (int)(y + ys/2);
    m->z = (int)(y + zs/2);
    m->dx = m->dy = m->dz = -1;

    m->dx0 = frag->dx;
    m->dy0 = frag->dy;
    m->dz0 = frag->dz;

    m->start = frag->start;
    
    m->end = -1;
}

__device__ void ini(hforces::Frag *frag,
                     float x, float y, float z,
                     /**/ M *m) {
    ini0(frag, x, y, z, XS, YS, ZS, /**/ m);
}

__device__ int nxt_xyz(int *pdx, int *pdy, int *pdz) {
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

__device__ int d2i0(int dx0, int dy0, int dz0,
                    int x, int y, int z,
                    int xs, int ys, int zs) {
    if (dx0 != 0) {
        dx0 = dy0; x = y; xs = ys;
        dy0 = dz0; y = z; ys = zs;
    } else if (dy0 != 0) {
        dy0 = dz0; y = z; ys = zs;
    }
    if (dx0 != 0) {x = 0; xs = 1;}
    if (dy0 != 0) {y = 0;        }
    return y * xs  + x;
}

__device__ int d2i(int dx0, int dy0, int dz0,
                   int x, int y, int z) {
    return d2i0(dx0, dy0, dz0, x, y, z, XS, YS, ZS);
}

__device__ int nxt0(/**/ M* m, int *pi, int *pj,
                    int xs, int ys, int zs) {
    int i, j, end;
    int dx, dy, dz, dx0, dy0, dz0, x, y, z;
    const int *start;
    i = m->i; end = m->end;
    if (end != -1 && ++i < end) goto old;
    dx  = m->dx;   dy = m->dy;   dz = m->dz;
    dx0 = m->dx0; dy0 = m->dy0; dz0 = m->dz0;
    for (;;) {
        if (nxt_xyz(&dx, &dy, &dz) == END) goto end;
        if (valid(dx0, dy0, dz0, dx, dy, dz))
            goto new0;
    }
 end:
    return END;
 old:
    m->i = *pi = i; return OLD;
 new0:
    x = m->x; y = m->y; z = m->z;
    x -= xs * dx0; y -= ys * dy0; z -= zs * dz0;
    j = d2i0(dx0, dy0, dz0,   x, y, z,   xs, ys, zs);
    start = m->start;
    m->i = *pi = start[j];
    m->end =     start[j + 1];
    m->dx = dx; m->dy = dy; m->dz = dz;
    *pj = j;
    return NEW;
}

__device__ int nxt(/**/ M* m, int *pi, int *pj) {
    return nxt0(m, pi, pj, XS, YS, ZS);
}

} /* namespace */
