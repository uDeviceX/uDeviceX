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

template <typename Pa>
_S_ void add_to_grid(const Pa *p, const Grid *g) {
    int3 gcoords;
    int gid;
    gcoords = get_cell_coords(p->r, g->L, g->N);
    gid = get_grid_id(gcoords, g->N);
    add_part(gid, p, g);
}

template <typename Datum>
__global__ void add(const Datum data, /**/ Grid grid) {
    int pid;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= data.n) return;

    auto p = fetch_part(pid, &data);
    add_to_grid(&p, &grid);    
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

__global__ void avg(int nsteps, Grid g) {
    long i;
    float inv_rho, inv_nsteps;
    float rho, vol;
    float sv, sr, ss; /* scales for v, rho, stress */
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= get_size(&g)) return;

    rho = g.d[RHO][i];
    vol = cell_volume(&g);
        
    inv_rho    = fabs(rho) > 1e-6 ? 1.f / rho : 0;
    inv_nsteps = 1.0 / nsteps;

    sv = inv_rho * inv_nsteps;
    sr = inv_nsteps / vol;
    
    g.d[RHO][i] = sr * rho;
    g.d[VX][i] *= sv;
    g.d[VY][i] *= sv;
    g.d[VZ][i] *= sv;

    if (g.stress) {
        ss = 1.f / vol;

        g.d[SXX][i] *= ss;
        g.d[SXY][i] *= ss;
        g.d[SXZ][i] *= ss;
        g.d[SYY][i] *= ss;
        g.d[SYZ][i] *= ss;
        g.d[SZZ][i] *= ss;
    }
}

#undef _S_
