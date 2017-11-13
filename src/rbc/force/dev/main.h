/* particle - float2 union */
union Part {
    float2 f2[3];
    struct { float3 r, v; };
};

/* position - float2 union */
union Pos {
    float2 f2[2];
    struct { float3 r; float dummy; };
};

static __device__ Pos tex2Pos(const Texo<float2> vert, const int id) {
    Pos r;
    r.f2[0] = fetch(vert, 3 * id + 0);
    r.f2[1] = fetch(vert, 3 * id + 1);
    return r;
}

static __device__ Part tex2Part(const Texo<float2> vert, const int id) {
    Part p;
    p.f2[0] = fetch(vert, 3 * id + 0);
    p.f2[1] = fetch(vert, 3 * id + 1);
    p.f2[2] = fetch(vert, 3 * id + 2);
    return p;
}

static __device__ float3 adj_tris(int md, int nv, const Texo<float2> vert, const Texo<int> adj0,
                                  const Part p0, const float *av) {
    int pid, lid, idrbc, offset, neighid, i1, i2;
    float3 f, fv;
    bool valid;

    pid     = (threadIdx.x + blockDim.x * blockIdx.x) / md;
    neighid = (threadIdx.x + blockDim.x * blockIdx.x) % md;
    lid   = pid % nv;
    idrbc = pid / nv;
    offset = idrbc * nv;
    i1 = fetch(adj0, neighid + md * lid);
    valid = i1 != -1;

    i2 = fetch(adj0, ((neighid + 1) % md) + md * lid);
    if (i2 == -1 && valid)
        i2 = fetch(adj0, 0 + md * lid);

    if (valid) {
        const Part p1 = tex2Part(vert, offset + i1);
        const Pos  r2 = tex2Pos(vert, offset + i2);

        f  = tri(p0.r, p1.r, r2.r, av[2 * idrbc], av[2 * idrbc + 1]);
        fv = visc(p0.r, p1.r, p0.v, p1.v);
        add(&fv, /**/ &f);
        return f;
    }
    return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
}

static __device__ float3 adj_dihedrals(int md, int nv, const Texo<float2> vert, const Texo<int> adj0,
                                       const Texo<int> adj1, float3 r0) {
    int pid, lid, offset, neighid;
    int i1, i2, i3, i4;
    bool valid;

    pid     = (threadIdx.x + blockDim.x * blockIdx.x) / md;
    neighid = (threadIdx.x + blockDim.x * blockIdx.x) % md;

    offset = (pid / nv) * nv;
    lid =     pid % nv;

    /*
      r4
      /   \
      r1 --> r2 --> r3
      \   /
      V
      r0

      dihedrals: 0124, 0123
    */

    i1 = fetch(adj0, neighid + md * lid);
    valid = i1 != -1;

    i2 = fetch(adj0, ((neighid + 1) % md) + md * lid);

    if (i2 == -1 && valid) {
        i2 = fetch(adj0, 0 + md * lid);
        i3 = fetch(adj0, 1 + md * lid);
    } else {
        i3 = fetch(adj0, ((neighid + 2) % md) + md * lid);
        if (i3 == -1 && valid) i3 = fetch(adj0, 0 + md * lid);
    }
    i4 = fetch(adj1, neighid + md * lid);

    if (valid) {
        const Pos r1 = tex2Pos(vert, offset + i1);
        const Pos r2 = tex2Pos(vert, offset + i2);
        const Pos r3 = tex2Pos(vert, offset + i3);
        const Pos r4 = tex2Pos(vert, offset + i4);
        float3 fd1, fd2;
        fd1 = dihedral<1>(r0, r2.r, r1.r, r4.r);
        fd2 = dihedral<2>(r1.r, r0, r2.r, r3.r);
        add(&fd1, /**/ &fd2);
        return fd2;
    }
    return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
}

static __global__ void force(int md, int nv, const Texo<float2> vert, const Texo<int> adj0, const Texo<int> adj1,
                             int nc, const float *__restrict__ av, float *ff) {
    int i, pid;
    float3 f, fd;
    Map m;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    pid = i / md;
    
    if (pid < nc * nv) {
        ini_map(md, nv, i, adj0, adj1, /**/ &m);

        const Part p0 = tex2Part(vert, pid);

        /* all triangles and dihedrals adjusting to vertex `pid` */
        f  = adj_tris(md, nv, vert, adj0, p0, av);
        fd = adj_dihedrals(md, nv, vert, adj0, adj1, p0.r);
        add(&fd, /**/ &f);

        if (f.x > -1.0e9f) {
            atomicAdd(&ff[3 * pid + 0], f.x);
            atomicAdd(&ff[3 * pid + 1], f.y);
            atomicAdd(&ff[3 * pid + 2], f.z);
        }
    }
}
