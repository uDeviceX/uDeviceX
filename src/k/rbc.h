namespace k_rbc
{

texture<float2, 1, cudaReadModeElementType> Vert; /* vertices */
texture<int, 1, cudaReadModeElementType> Adj0;    /* adjacency lists */
texture<int, 1, cudaReadModeElementType> Adj1;
texture<int4, cudaTextureType1D> Tri;

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

__device__ float3 angle0(float2 t0, float2 t1, float *av) {
    int degreemax, pid, lid, idrbc, offset, neighid, idv2, idv3;
    float2 t2, t3, t4;
    float3 v1, u1, v2, u2, v3, f;
    bool valid;

    degreemax = 7; /* :TODO: duplicate */
    pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
    lid = pid % RBCnv;
    idrbc = pid / RBCnv;
    offset = idrbc * RBCnv * 3;
    neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

    t2 = tex1Dfetch(Vert, pid * 3 + 2);
    v1 = make_float3(t0.x, t0.y, t1.x);
    u1 = make_float3(t1.y, t2.x, t2.y);

    idv2 = tex1Dfetch(Adj0, neighid + degreemax * lid);
    valid = idv2 != -1;

    idv3 = tex1Dfetch(Adj0, ((neighid + 1) % degreemax) + degreemax * lid);


    if (idv3 == -1 && valid) idv3 = tex1Dfetch(Adj0, 0 + degreemax * lid);

    if (valid) {
	t0 = tex1Dfetch(Vert, offset + idv2 * 3 + 0);
	t1 = tex1Dfetch(Vert, offset + idv2 * 3 + 1);
	t2 = tex1Dfetch(Vert, offset + idv2 * 3 + 2);
	t3 = tex1Dfetch(Vert, offset + idv3 * 3 + 0);
	t4 = tex1Dfetch(Vert, offset + idv3 * 3 + 1);

	v2 = make_float3(t0.x, t0.y, t1.x);
	u2 = make_float3(t1.y, t2.x, t2.y);
	v3 = make_float3(t3.x, t3.y, t4.x);

	f  = angle(v1, v2, v3, av[2 * idrbc], av[2 * idrbc + 1]);
	f += visc(v1, v2, u1, u2);
	return f;
    }
    return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
}

__device__ float3 dihedral(float2 t0, float2 t1) {
    int degreemax, pid, lid, offset, neighid;
    int idv1, idv2, idv3, idv4;
    float2         t2, t3, t4, t5, t6, t7;
    float3 v0, v1, v2, v3, v4;
    bool valid;

    degreemax = 7;
    pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
    lid = pid % RBCnv;
    offset = (pid / RBCnv) * RBCnv * 3;
    neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

    v0 = make_float3(t0.x, t0.y, t1.x);

    /*
      v4
      /   \
      v1 --> v2 --> v3
      \   /
      V
      v0

      dihedrals: 0124, 0123
    */


    idv1 = tex1Dfetch(Adj0, neighid + degreemax * lid);
    valid = idv1 != -1;

    idv2 = tex1Dfetch(Adj0, ((neighid + 1) % degreemax) + degreemax * lid);

    if (idv2 == -1 && valid) {
	idv2 = tex1Dfetch(Adj0, 0 + degreemax * lid);
	idv3 = tex1Dfetch(Adj0, 1 + degreemax * lid);
    } else {
	idv3 =
	    tex1Dfetch(Adj0, ((neighid + 2) % degreemax) + degreemax * lid);
	if (idv3 == -1 && valid) idv3 = tex1Dfetch(Adj0, 0 + degreemax * lid);
    }

    idv4 = tex1Dfetch(Adj1, neighid + degreemax * lid);

    if (valid) {
	t0 = tex1Dfetch(Vert, offset + idv1 * 3 + 0);
	t1 = tex1Dfetch(Vert, offset + idv1 * 3 + 1);
	t2 = tex1Dfetch(Vert, offset + idv2 * 3 + 0);
	t3 = tex1Dfetch(Vert, offset + idv2 * 3 + 1);
	t4 = tex1Dfetch(Vert, offset + idv3 * 3 + 0);
	t5 = tex1Dfetch(Vert, offset + idv3 * 3 + 1);
	t6 = tex1Dfetch(Vert, offset + idv4 * 3 + 0);
	t7 = tex1Dfetch(Vert, offset + idv4 * 3 + 1);

	v1 = make_float3(t0.x, t0.y, t1.x);
	v2 = make_float3(t2.x, t2.y, t3.x);
	v3 = make_float3(t4.x, t4.y, t5.x);
	v4 = make_float3(t6.x, t6.y, t7.x);

	return dihedral0<1>(v0, v2, v1, v4) + dihedral0<2>(v1, v0, v2, v3);
    }
    return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
}

__global__ void force(int nc, float *__restrict__ av,
			    float *acc) {
    int degreemax = 7;
    int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;

    if (pid < nc * RBCnv) {
	float2 t0 = tex1Dfetch(Vert, pid * 3 + 0);
	float2 t1 = tex1Dfetch(Vert, pid * 3 + 1);

	float3 f = angle0(t0, t1, av);
	f += dihedral(t0, t1);

	if (f.x > -1.0e9f) {
	    atomicAdd(&acc[3 * pid + 0], f.x);
	    atomicAdd(&acc[3 * pid + 1], f.y);
	    atomicAdd(&acc[3 * pid + 2], f.z);
	}
    }
}

__DF__ float3 tex2vec(int id) {
    float2 t0 = tex1Dfetch(Vert, id + 0);
    float2 t1 = tex1Dfetch(Vert, id + 1);
    return make_float3(t0.x, t0.y, t1.x);
}

__DF__ float2 warpReduceSum(float2 val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
	val.x += __shfl_down(val.x, offset);
	val.y += __shfl_down(val.y, offset);
    }
    return val;
}

__global__ void area_volume(float *totA_V) {
    float2 a_v = make_float2(0.0f, 0.0f);
    int cid = blockIdx.y;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < RBCnt;
	 i += blockDim.x * gridDim.x) {
	int4 ids = tex1Dfetch(Tri, i);

	float3 v0(tex2vec(3 * (ids.x + cid * RBCnv)));
	float3 v1(tex2vec(3 * (ids.y + cid * RBCnv)));
	float3 v2(tex2vec(3 * (ids.z + cid * RBCnv)));

	a_v.x += area0(v0, v1, v2);
	a_v.y += volume0(v0, v1, v2);
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
