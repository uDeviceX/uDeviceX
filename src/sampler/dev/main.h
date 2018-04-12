#define _S_ static __device__

_S_ void fetch_part(int i, const SampleDatum *d, Part *p) {
    float2 p0, p1, p2;
    const float2 *pp = (const float2*) d->pp;
    p0 = pp[3*i+0]; p1 = pp[3*i+1]; p2 = pp[3*i+2];

    p->r = make_float3(p0.x, p0.y, p1.x);
    p->v = make_float3(p1.y, p2.x, p2.y);
}

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

_S_ void add_to_grid(const Part *p, Grid g) {
    int3 gcoords;
    float4 *dst;
    int gid;
    gcoords = get_cell_coords(p->r, g.L, g.N);
    gid = get_grid_id(gcoords, g.N);
    dst = g.pp + gid;
    
    atomicAdd(&dst->x, p->v.x);
    atomicAdd(&dst->y, p->v.x);
    atomicAdd(&dst->z, p->v.x);
    atomicAdd(&dst->w, 1.f);
}

__global__ void add(const SampleDatum data, /**/ Grid grid) {
    int pid;    
    Part p;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= data.n) return;

    fetch_part(pid, &data, /**/ &p);
    add_to_grid(&p, grid);    
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
    float4 d;
    float s, st;    
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= get_size(&g)) return;

    d = g.pp[i];

    st = 1.0 / nsteps;
    s = d.x ? 0 : st / d.w;

    d.x *= s;
    d.y *= s;
    d.z *= s;
    d.w *= st / cell_volume(&g);

    g.pp[i] = d;
}

#undef _S_
