struct Fi { /* field */
    const float *org, *spa;
    const int *n;
    const float *D;
};

static void fi_ini(const float *org, const float *spa, const int *n, const float *D,
                   /**/ Fi *fi) {
    fi->org = org; fi->spa = spa; fi->n = n; fi->D = D;
}

static void fi_r(const Fi *fi, int ix, int iy, int iz, /**/ float *r) {
    enum {X, Y, Z};
    const float *org, *spa;
    org = fi->org; spa = fi->spa;
    r[X] = org[X] + (ix + 0.5) * spa[X] - 0.5;
    r[Y] = org[Y] + (iy + 0.5) * spa[Y] - 0.5;
    r[Z] = org[Z] + (iz + 0.5) * spa[Z] - 0.5;
}

static float fi_get(const Fi *fi, int ix, int iy, int iz) {
    enum {X, Y};
    const float *D;
    const int *n;
    int i;
    D = fi->D; n = fi->n;
    i = ix + n[X] * (iy + n[Y] * iz);
    return D[i];
}

static float spl(float x) { /* b-spline (see poc/spline/main.mac) */
    return
        x <= 0 ? 0.0 :
        x <= 1 ? x*x*x/6 :
        x <= 2 ? (x*((12-3*x)*x-12)+4)/6 :
        x <= 3 ? (x*(x*(3*x-24)+60)-44)/6 :
        x <= 4 ? (x*((12-x)*x-48)+64)/6 :
        0.0;
}

void sample(const float org[3], const float spa[3], const int N0[3], const float *D0, const int N1[3], float *D1) {
    /* org: origin, spa: spacing, N[01]: number of points; D[01]: data
       sample from grid `0' to `1'
       org, spa: are for `0'
    */
    Fi fi;
    fi_ini(org, spa, N0, D0, /**/ &fi);
    enum {X, Y, Z};
#define OOO(ix, iy, iz) (D1 [ix + N1[X] * (iy + N1[Y] * iz)])
    int iz, iy, ix, i, c, sx, sy, sz, anchor[3], g[3];
    float val, s, r[3], w[3][4], tmp[4][4], partial[4];
    for (iz = 0; iz < N1[Z]; ++iz)
        for (iy = 0; iy < N1[Y]; ++iy)
            for (ix = 0; ix < N1[X]; ++ix) {
                fi_r(&fi, ix, iy, iz, /**/ r);
                for (c = 0; c < 3; ++c) anchor[c] = (int)floor(r[c]);
                for (c = 0; c < 3; ++c)
                    for (i = 0; i < 4; ++i)
                        w[c][i] = spl(r[c] - (anchor[c] - 1 + i) + 2);
                for (sz = 0; sz < 4; ++sz)
                    for (sy = 0; sy < 4; ++sy) {
                        s = 0;
                        for (sx = 0; sx < 4; ++sx) {
                            int l[3] = {sx, sy, sz};
                            for (c = 0; c < 3; ++c)
                                g[c] = (l[c] - 1 + anchor[c] + N0[c]) % N0[c];
                            s += w[0][sx] * fi_get(&fi, g[X], g[Y], g[Z]);
                        }
                        tmp[sz][sy] = s;
                    }
                for (sz = 0; sz < 4; ++sz) {
                    s = 0;
                    for (sy = 0; sy < 4; ++sy) s += w[1][sy] * tmp[sz][sy];
                    partial[sz] = s;
                }
                val = 0;
                for (sz = 0; sz < 4; ++sz) val += w[2][sz] * partial[sz];
                OOO(ix, iy, iz) = val;
            }
#undef OOO
}
