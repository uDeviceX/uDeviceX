namespace field {
void ini(const char *path, int N[3], float extent[3], float* grid_data) { /* read sdf file */
    FILE *fh = fopen(path, "r");
    char line[2048];
    fgets(line, sizeof(line), fh);
    sscanf(line, "%f %f %f", &extent[0], &extent[1], &extent[2]);
    fgets(line, sizeof(line), fh);
    sscanf(line, "%d %d %d", &N[0], &N[1], &N[2]);

    assert(N[0]*N[1]*N[2] <= MAX_SUBDOMAIN_VOLUME);
    
    MC(MPI_Bcast(N, 3, MPI_INT, 0, m::cart));
    MC(MPI_Bcast(extent, 3, MPI_FLOAT, 0, m::cart));

    int np = N[0] * N[1] * N[2];
    fread(grid_data, sizeof(float), np, fh);
    fclose(fh);
    MPI_Barrier(m::cart);
}

void sample(float rlo[3], float dr[3], int nsize[3], int N[3], float ampl, float* grid_data, float *out) {
    enum {X, Y, Z};
#define OOO(ix, iy, iz) (      out[ix + nsize[X] * (iy + nsize[Y] * iz)])
#define DDD(ix, iy, iz) (grid_data [ix +     N[X] * (iy +     N[Y] * iz)])
#define i2r(i, d) (rlo[d] + (i + 0.5) * dr[d] - 0.5)
#define i2x(i)    i2r(i,X)
#define i2y(i)    i2r(i,Y)
#define i2z(i)    i2r(i,Z)
    Bspline<4> bsp;
    int iz, iy, ix, i, c, sx, sy, sz;
    float s;
    for (iz = 0; iz < nsize[Z]; ++iz)
    for (iy = 0; iy < nsize[Y]; ++iy)
	for (ix = 0; ix < nsize[X]; ++ix) {
        float r[3] = {(float) i2x(ix), (float) i2y(iy), (float) i2z(iz)};

        int anchor[3];
        for (c = 0; c < 3; ++c) anchor[c] = (int)floor(r[c]);

        float w[3][4];
        for (c = 0; c < 3; ++c)
	    for (i = 0; i < 4; ++i)
        w[c][i] = bsp.eval<0>(r[c] - (anchor[c] - 1 + i) + 2);

        float tmp[4][4];
        for (sz = 0; sz < 4; ++sz)
	    for (sy = 0; sy < 4; ++sy) {
            s = 0;
            for (sx = 0; sx < 4; ++sx) {
                int l[3] = {sx, sy, sz};
                int g[3];
                for (c = 0; c < 3; ++c)
                g[c] = (l[c] - 1 + anchor[c] + N[c]) % N[c];

                s += w[0][sx] * DDD(g[X], g[Y], g[Z]);
            }
            tmp[sz][sy] = s;
	    }
        float partial[4];
        for (sz = 0; sz < 4; ++sz) {
            s = 0;
            for (sy = 0; sy < 4; ++sy) s += w[1][sy] * tmp[sz][sy];
            partial[sz] = s;
        }
        float val = 0;
        for (sz = 0; sz < 4; ++sz) val += w[2][sz] * partial[sz];
        OOO(ix, iy, iz) = val * ampl;
	}
#undef DDD
#undef OOO
}

void dump0(int N[3], float extent[3], float* grid_data, float* walldata) {
    int c, L[3] = {XS, YS, ZS};
    float rlo[3], dr[3], ampl;
    for (c = 0; c < 3; ++c) {
        rlo[c] = m::coords[c] * L[c] / (float)(m::dims[c] * L[c]) * N[c];
        dr[c] = N[c] / (float)(m::dims[c] * L[c]);
    }
    ampl = L[0] / (extent[0] / (float) m::dims[0]);
    field::sample(rlo, dr, L, N, ampl, grid_data, walldata);
    H5FieldDump dump;
    dump.dump_scalarfield(walldata, "wall");
}

void dump(int N[], float extent[], float* grid_data) {
    float *walldata = (float*)malloc(sizeof(walldata[0])*MAX_SUBDOMAIN_VOLUME);
    dump0(N, extent, grid_data, walldata);
    free(walldata);
}
} /* namespace field */
