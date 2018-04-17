#define _S_ static __device__

_S_ float get_spacing(int L, int N) {
    return (float) L / (float) N;
}

_S_ int get_coord(float x, int L, int N) {
    int i;
    float dx;
    dx = get_spacing(L, N);
    i = (x + L/2) / dx;
    return min(N-1, max(0, i));
}

_S_ int3 get_cell_coords(float3 r, int3 L, int3 N) {
    return make_int3(get_coord(r.x, L.x, N.x),
                     get_coord(r.y, L.y, N.y),
                     get_coord(r.z, L.z, N.z));                     
}

_S_ int get_grid_id(int3 gc, int3 N) {
    return gc.x + N.x * (gc.y + N.y * gc.z);
}

_S_ int get_grid_id(const Part *p, const Grid *g) {
    int3 gcoords;
    gcoords = get_cell_coords(p->r, g->L, g->N);
    return get_grid_id(gcoords, g->N);
}

__global__ void add(SampleDatum data, /**/ Grid grid) {
    int pid, gid; /* particle and grid id */
    Part p;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= data.n) return;

    fetch_part(pid, &data, &p);
    gid = get_grid_id(&p, &grid);

    add_part(gid, &p, &grid);

    if (grid.stress) {
        Stress s;
        fetch_stress(pid, &data, &s);
        add_stress(gid, &s, &grid);
    }
}

_S_ long get_size(const Grid *g) {
    int3 N = g->N;
    return N.x * N.y * N.z;
}

_S_ float cell_volume(const Grid *g) {
    float dx, dy, dz;
    dx = get_spacing(g->L.x, g->N.x);
    dy = get_spacing(g->L.y, g->N.y);
    dz = get_spacing(g->L.z, g->N.z);
    return dx * dy * dz;
}

__global__ void space_avg(Grid g) {
    long i;
    float inv_rho;
    float rho, vol;
    float sv, sr, ss; /* scales for v, rho, stress */
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= get_size(&g)) return;

    rho = g.p[RHO][i];
    vol = cell_volume(&g);
        
    inv_rho    = fabs(rho) > 1e-6 ? 1.f / rho : 0;

    sv = inv_rho;
    sr = 1.f / vol;
    
    g.p[RHO][i] = sr * rho;

    g.p[VX][i] *= sv;
    g.p[VY][i] *= sv;
    g.p[VZ][i] *= sv;

    if (g.stress) {
        ss = sr;
        g.s[SXX][i] *= ss;
        g.s[SXY][i] *= ss;
        g.s[SXZ][i] *= ss;
        g.s[SYY][i] *= ss;
        g.s[SYZ][i] *= ss;
        g.s[SZZ][i] *= ss;
    }
}

__global__ void add_to_grid(const Grid src, Grid dst) {
    long i, j;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= get_size(&src)) return;

    for (j = 0; j < NFIELDS_P; ++j)
        dst.p[j][i] += src.p[j][i];

    if (src.stress) {
        for (j = 0; j < NFIELDS_S; ++j)
            dst.s[j][i] += src.s[j][i];
    }
}

__global__ void time_avg(int nsteps, Grid g) {
    long i, j;
    float s;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= get_size(&g)) return;

    s = 1.0 / nsteps;

    for (j = 0; j < NFIELDS_P; ++j)
        g.p[j][i] *= s;

    if (g.stress)
        for (j = 0; j < NFIELDS_S; ++j)
            g.s[j][i] *= s;

}

#undef _S_
