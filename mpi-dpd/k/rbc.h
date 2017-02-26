namespace CudaRBC {
static __constant__ Params devParams;
texture<float2, 1, cudaReadModeElementType> texVertices;
texture<int, 1, cudaReadModeElementType> texAdjVert;
texture<int, 1, cudaReadModeElementType> texAdjVert2;
texture<int4, cudaTextureType1D> texTriangles4;
__constant__ float A[4][4];

__device__ __forceinline__ float3 _fangle(float3 v1, float3 v2,
					  float3 v3, float area,
					  float volume) {
  float3 x21 = v2 - v1;
  float3 x32 = v3 - v2;
  float3 x31 = v3 - v1;

  float3 normal = cross(x21, x31);

  float Ak = 0.5 * sqrtf(dot(normal, normal));
  float A0 = devParams.Area0;
  float n_2 = 1.0 / Ak;
  float coefArea =
      -0.25f * (devParams.ka * (area - devParams.totArea0) * n_2) -
      devParams.kd * (Ak - A0) / (4. * A0 * Ak);

  float coeffVol = devParams.kv * (volume - devParams.totVolume0);
  float3 addFArea = coefArea * cross(normal, x32);
  float3 addFVolume = coeffVol * cross(v3, v2);

  float r = length(v2 - v1);
  r = r < 0.0001f ? 0.0001f : r;
  float xx = r / devParams.lmax;
  float IbforceI_wcl =
      devParams.kbToverp * (0.25f / ((1.0f - xx) * (1.0f - xx)) - 0.25f + xx) /
      r;
  float kp = devParams.kp;
  float mpow = devParams.mpow;
  float IbforceI_pow = -kp / powf(r, mpow) / r;

  return addFArea + addFVolume + (IbforceI_wcl + IbforceI_pow) * x21;
}

__device__ __forceinline__ float3 _fvisc(float3 v1, float3 v2,
					 float3 u1, float3 u2) {
  float3 du = u2 - u1;
  float3 dr = v1 - v2;

  return du * devParams.gammaT +
	 dr * devParams.gammaC * dot(du, dr) / dot(dr, dr);
}

template <int update>
__device__ __forceinline__ float3 _fdihedral(float3 v1, float3 v2, float3 v3,
					     float3 v4) {
  float3 ksi = cross(v1 - v2, v1 - v3);
  float3 dzeta = cross(v3 - v4, v2 - v4);

  float overIksiI = rsqrtf(dot(ksi, ksi));
  float overIdzetaI = rsqrtf(dot(dzeta, dzeta));

  float cosTheta = dot(ksi, dzeta) * overIksiI * overIdzetaI;
  float IsinThetaI2 = 1.0f - cosTheta * cosTheta;
  float sinTheta_1 = copysignf(
      rsqrtf(max(IsinThetaI2, 1.0e-6f)),
      dot(ksi - dzeta, v4 - v1)); // ">" because the normals look inside
  float beta =
      devParams.cost0kb - cosTheta * devParams.sint0kb * sinTheta_1;

  float b11 = -beta * cosTheta * overIksiI * overIksiI;
  float b12 = beta * overIksiI * overIdzetaI;
  float b22 = -beta * cosTheta * overIdzetaI * overIdzetaI;

  if (update == 1)
    return cross(ksi, v3 - v2) * b11 + cross(dzeta, v3 - v2) * b12;
  else if (update == 2)
    return cross(ksi, v1 - v3) * b11 +
	   (cross(ksi, v3 - v4) + cross(dzeta, v1 - v3)) * b12 +
	   cross(dzeta, v3 - v4) * b22;
  else
    return make_float3(0, 0, 0);
}

template <int nvertices>
__device__ float3 _fangle_device(float2 tmp0, float2 tmp1,
				 float *av) {
  int degreemax = 7;
  int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
  int lid = pid % nvertices;
  int idrbc = pid / nvertices;
  int offset = idrbc * nvertices * 3;
  int neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

  float2 tmp2 = tex1Dfetch(texVertices, pid * 3 + 2);
  float3 v1 = make_float3(tmp0.x, tmp0.y, tmp1.x);
  float3 u1 = make_float3(tmp1.y, tmp2.x, tmp2.y);

  int idv2 = tex1Dfetch(texAdjVert, neighid + degreemax * lid);
  bool valid = idv2 != -1;

  int idv3 =
      tex1Dfetch(texAdjVert, ((neighid + 1) % degreemax) + degreemax * lid);

  if (idv3 == -1 && valid) idv3 = tex1Dfetch(texAdjVert, 0 + degreemax * lid);

  if (valid) {
    float2 tmp0 = tex1Dfetch(texVertices, offset + idv2 * 3 + 0);
    float2 tmp1 = tex1Dfetch(texVertices, offset + idv2 * 3 + 1);
    float2 tmp2 = tex1Dfetch(texVertices, offset + idv2 * 3 + 2);
    float2 tmp3 = tex1Dfetch(texVertices, offset + idv3 * 3 + 0);
    float2 tmp4 = tex1Dfetch(texVertices, offset + idv3 * 3 + 1);

    float3 v2 = make_float3(tmp0.x, tmp0.y, tmp1.x);
    float3 u2 = make_float3(tmp1.y, tmp2.x, tmp2.y);
    float3 v3 = make_float3(tmp3.x, tmp3.y, tmp4.x);

    float3 f = _fangle(v1, v2, v3, av[2 * idrbc], av[2 * idrbc + 1]);
    f += _fvisc(v1, v2, u1, u2);
    return f;
  }
  return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
}

template <int nvertices>
__device__ float3 _fdihedral_device(float2 tmp0, float2 tmp1) {
  int degreemax = 7;
  int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
  int lid = pid % nvertices;
  int offset = (pid / nvertices) * nvertices * 3;
  int neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

  float3 v0 = make_float3(tmp0.x, tmp0.y, tmp1.x);

  /*
	 v4
       /   \
     v1 --> v2 --> v3
       \   /
	 V
	 v0

   dihedrals: 0124, 0123
  */

  int idv1, idv2, idv3, idv4;
  idv1 = tex1Dfetch(texAdjVert, neighid + degreemax * lid);
  bool valid = idv1 != -1;

  idv2 = tex1Dfetch(texAdjVert, ((neighid + 1) % degreemax) + degreemax * lid);

  if (idv2 == -1 && valid) {
    idv2 = tex1Dfetch(texAdjVert, 0 + degreemax * lid);
    idv3 = tex1Dfetch(texAdjVert, 1 + degreemax * lid);
  } else {
    idv3 =
	tex1Dfetch(texAdjVert, ((neighid + 2) % degreemax) + degreemax * lid);
    if (idv3 == -1 && valid) idv3 = tex1Dfetch(texAdjVert, 0 + degreemax * lid);
  }

  idv4 = tex1Dfetch(texAdjVert2, neighid + degreemax * lid);

  if (valid) {
    float2 tmp0 = tex1Dfetch(texVertices, offset + idv1 * 3 + 0);
    float2 tmp1 = tex1Dfetch(texVertices, offset + idv1 * 3 + 1);
    float2 tmp2 = tex1Dfetch(texVertices, offset + idv2 * 3 + 0);
    float2 tmp3 = tex1Dfetch(texVertices, offset + idv2 * 3 + 1);
    float2 tmp4 = tex1Dfetch(texVertices, offset + idv3 * 3 + 0);
    float2 tmp5 = tex1Dfetch(texVertices, offset + idv3 * 3 + 1);
    float2 tmp6 = tex1Dfetch(texVertices, offset + idv4 * 3 + 0);
    float2 tmp7 = tex1Dfetch(texVertices, offset + idv4 * 3 + 1);

    float3 v1 = make_float3(tmp0.x, tmp0.y, tmp1.x);
    float3 v2 = make_float3(tmp2.x, tmp2.y, tmp3.x);
    float3 v3 = make_float3(tmp4.x, tmp4.y, tmp5.x);
    float3 v4 = make_float3(tmp6.x, tmp6.y, tmp7.x);

    return _fdihedral<1>(v0, v2, v1, v4) + _fdihedral<2>(v1, v0, v2, v3);
  }
  return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
}

template <int nvertices>
__global__ void fall_kernel(int nrbcs, float *__restrict__ av,
			    float *acc) {
  int degreemax = 7;
  int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;

  if (pid < nrbcs * nvertices) {
    float2 tmp0 = tex1Dfetch(texVertices, pid * 3 + 0);
    float2 tmp1 = tex1Dfetch(texVertices, pid * 3 + 1);

    float3 f = _fangle_device<nvertices>(tmp0, tmp1, av);
    f += _fdihedral_device<nvertices>(tmp0, tmp1);

    if (f.x > -1.0e9f) {
      atomicAdd(&acc[3 * pid + 0], f.x);
      atomicAdd(&acc[3 * pid + 1], f.y);
      atomicAdd(&acc[3 * pid + 2], f.z);
    }
  }
}

 __global__ void addKernel(float* axayaz, float* __restrict__ addfrc, int n)
{
  uint pid = threadIdx.x + blockIdx.x * blockDim.x;
  if (pid < n) axayaz[3*pid + 0] += addfrc[pid];
}

__device__ __forceinline__ float3 tex2vec(int id) {
  float2 tmp0 = tex1Dfetch(texVertices, id + 0);
  float2 tmp1 = tex1Dfetch(texVertices, id + 1);
  return make_float3(tmp0.x, tmp0.y, tmp1.x);
}

__device__ __forceinline__ float2 warpReduceSum(float2 val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
  }
  return val;
}

__global__ void areaAndVolumeKernel(float *totA_V) {
  float2 a_v = make_float2(0.0f, 0.0f);
  int cid = blockIdx.y;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < devParams.ntriangles;
       i += blockDim.x * gridDim.x) {
    int4 ids = tex1Dfetch(texTriangles4, i);

    float3 v0(tex2vec(3 * (ids.x + cid * devParams.nvertices)));
    float3 v1(tex2vec(3 * (ids.y + cid * devParams.nvertices)));
    float3 v2(tex2vec(3 * (ids.z + cid * devParams.nvertices)));

    a_v.x += 0.5f * length(cross(v1 - v0, v2 - v0));
    a_v.y += 0.1666666667f *
	     (-v0.z * v1.y * v2.x + v0.z * v1.x * v2.y + v0.y * v1.z * v2.x -
	      v0.x * v1.z * v2.y - v0.y * v1.x * v2.z + v0.x * v1.y * v2.z);
  }

  a_v = warpReduceSum(a_v);
  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicAdd(&totA_V[2 * cid + 0], a_v.x);
    atomicAdd(&totA_V[2 * cid + 1], a_v.y);
  }
}

}
