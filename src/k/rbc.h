namespace k_rbc
{

texture<float2, 1, cudaReadModeElementType> Vert; /* vertices */
texture<int, 1, cudaReadModeElementType> Adj0;    /* adjacency lists */
texture<int, 1, cudaReadModeElementType> Adj1;

/* first and second */
#define fst(t) ( (t).x )
#define scn(t) ( (t).y )

__device__ void tt2r(float2 t1, float2 t2, /**/ float3 *r) {
    r->x = fst(t1); r->y = scn(t1); r->z = fst(t2);
}

__device__ void ttt2ru(float2 t1, float2 t2, float2 t3, /**/ float3 *r, float3 *u) {
    r->x = fst(t1); r->y = scn(t1); r->z = fst(t2);
    u->x = scn(t2); u->y = fst(t3); u->z = scn(t3);
}

__device__ float3 adj_tris(float2 t0a, float2 t0b, float *av) {
    int nv = RBCnv;
    int degreemax, pid, lid, idrbc, offset, neighid, i2, i3;
    float2 t0c;
    float2 t1a, t1b, t1c, t2a, t2b;
    float3 r1, u1, r2, u2, r3, f;
    bool valid;

    degreemax = 7; /* :TODO: duplicate */
    pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
    lid = pid % nv;
    idrbc = pid / nv;
    offset = idrbc * nv * 3;
    neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;
    i2 = tex1Dfetch(Adj0, neighid + degreemax * lid);
    valid = i2 != -1;

    i3 = tex1Dfetch(Adj0, ((neighid + 1) % degreemax) + degreemax * lid);
    if (i3 == -1 && valid) i3 = tex1Dfetch(Adj0, 0 + degreemax * lid);

    if (valid) {
        t0c = tex1Dfetch(Vert,         pid * 3 + 2);
        t1a = tex1Dfetch(Vert, offset + i2 * 3 + 0);
        t1b = tex1Dfetch(Vert, offset + i2 * 3 + 1);
        t1c = tex1Dfetch(Vert, offset + i2 * 3 + 2);
        t2a = tex1Dfetch(Vert, offset + i3 * 3 + 0);
        t2b = tex1Dfetch(Vert, offset + i3 * 3 + 1);

        ttt2ru( t0a, t0b, t0c, &r1, &u1);
        ttt2ru( t1a, t1b, t1c, &r2, &u2);
        tt2r  ( t2a, t2b,      &r3     );

        f  = tri(r1, r2, r3, av[2 * idrbc], av[2 * idrbc + 1]);
        f += visc(r1, r2, u1, u2);
        return f;
    }
    return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
}

__device__ float3 adj_dihedrals(float2 t0a, float2 t0b) {
    int nv = RBCnv;

    int degreemax, pid, lid, offset, neighid;
    int i1, i2, i3, i4;
    float2 t1a, t1b, t2a, t2b, t3a, t3b, t4a, t4b;
    float3 r0, r1, r2, r3, r4;
    bool valid;

    degreemax = 7;
    pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
    lid = pid % nv;
    offset = (pid / nv) * nv * 3;
    neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

    r0 = make_float3(t0a.x, t0a.y, t0b.x);

    /*
      r4
      /   \
      r1 --> r2 --> r3
      \   /
      V
      r0

      dihedrals: 0124, 0123
    */


    i1 = tex1Dfetch(Adj0, neighid + degreemax * lid);
    valid = i1 != -1;

    i2 = tex1Dfetch(Adj0, ((neighid + 1) % degreemax) + degreemax * lid);

    if (i2 == -1 && valid) {
        i2 = tex1Dfetch(Adj0, 0 + degreemax * lid);
        i3 = tex1Dfetch(Adj0, 1 + degreemax * lid);
    } else {
        i3 =
            tex1Dfetch(Adj0, ((neighid + 2) % degreemax) + degreemax * lid);
        if (i3 == -1 && valid) i3 = tex1Dfetch(Adj0, 0 + degreemax * lid);
    }

    i4 = tex1Dfetch(Adj1, neighid + degreemax * lid);

    if (valid) {
        t1a = tex1Dfetch(Vert, offset + i1 * 3 + 0);
        t1b = tex1Dfetch(Vert, offset + i1 * 3 + 1);
        t2a = tex1Dfetch(Vert, offset + i2 * 3 + 0);
        t2b = tex1Dfetch(Vert, offset + i2 * 3 + 1);
        t3a = tex1Dfetch(Vert, offset + i3 * 3 + 0);
        t3b = tex1Dfetch(Vert, offset + i3 * 3 + 1);
        t4a = tex1Dfetch(Vert, offset + i4 * 3 + 0);
        t4b = tex1Dfetch(Vert, offset + i4 * 3 + 1);

        tt2r(t1a, t1b, &r1);
        tt2r(t2a, t2b, &r2);
        tt2r(t3a, t3b, &r3);
        tt2r(t4a, t4b, &r4);

        return dihedral<1>(r0, r2, r1, r4) + dihedral<2>(r1, r0, r2, r3);
    }
    return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
}

__global__ void force(int nc, float *__restrict__ av, float *ff) {
    int nv = RBCnv;
    int degreemax = 7;
    int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;

    if (pid < nc * nv) {
        float2 t0 = tex1Dfetch(Vert, pid * 3 + 0);
        float2 t1 = tex1Dfetch(Vert, pid * 3 + 1);

        /* all triangles and dihedrals adjusting to vertex `pid` */
        float3 f = adj_tris(t0, t1, av);
        f += adj_dihedrals(t0, t1);

        if (f.x > -1.0e9f) {
            atomicAdd(&ff[3 * pid + 0], f.x);
            atomicAdd(&ff[3 * pid + 1], f.y);
            atomicAdd(&ff[3 * pid + 2], f.z);
        }
    }
}

__DF__ float3 tex2vec(int id) {
    float2 ta = tex1Dfetch(Vert, id + 0);
    float2 tb = tex1Dfetch(Vert, id + 1);
    return make_float3(ta.x, ta.y, tb.x);
}

__DF__ float2 warpReduceSum(float2 val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
    }
    return val;
}

__global__ void area_volume(const Texo<int4> textri, float *totA_V) {
    int nv = RBCnv, nt = RBCnt;
    float2 a_v = make_float2(0.0f, 0.0f);
    int cid = blockIdx.y;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nt;
         i += blockDim.x * gridDim.x) {
        int4 ids = textri.fetch(i);

        float3 r0(tex2vec(3 * (ids.x + cid * nv)));
        float3 r1(tex2vec(3 * (ids.y + cid * nv)));
        float3 r2(tex2vec(3 * (ids.z + cid * nv)));

        a_v.x += area0(r0, r1, r2);
        a_v.y += volume0(r0, r1, r2);
    }
    a_v = warpReduceSum(a_v);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&totA_V[2 * cid + 0], a_v.x);
        atomicAdd(&totA_V[2 * cid + 1], a_v.y);
    }
}
#undef fst
#undef scn

} /* namespace k_rbc */
