namespace k {namespace sdist {
  __constant__ RedistPart::PackBuffer pack_buffers[27];
  __constant__ RedistPart::UnpackBuffer unpack_buffers[27];
  __device__   int pack_count[27], pack_start_padded[28];
  __constant__ int unpack_start[28], unpack_start_padded[28];
  __device__ bool failed;

  int ntexparticles = 0;
  float2 * texparticledata;
  texture<float, cudaTextureType1D> texAllParticles;
  texture<float2, cudaTextureType1D> texAllParticlesFloat2;


  __global__ void setup() {
    if (threadIdx.x == 0) failed = false;
    if (threadIdx.x < 27) pack_count[threadIdx.x] = 0;
  }

  __global__ void scatter_halo_indices_pack(int np) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid < np) {
	float xp[3];
	for(int c = 0; c < 3; ++c)  xp[c] = tex1Dfetch(texAllParticles, 6 * pid + c);
	int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	int vcode[3];
	for(int c = 0; c < 3; ++c)
	  vcode[c] = (2 + (xp[c] >= -L[c]/2) + (xp[c] >= L[c]/2)) % 3;

	int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);
	if (code > 0) {
	  int entry = atomicAdd(pack_count + code, 1);
	  if (entry < pack_buffers[code].capacity)
	    pack_buffers[code].scattered_indices[entry] = pid;
	}
      }
  }

  __global__ void tiny_scan(int nparticles, int bulkcapacity, int *packsizes, bool *failureflag) {
    int tid = threadIdx.x;
    int myval = 0, mycount = 0;

    if (tid < 27) {
      myval = mycount = pack_count[threadIdx.x];
      if (tid > 0) packsizes[tid] = mycount;
      if (mycount > pack_buffers[tid].capacity) {
	failed = true;
	*failureflag = true;
      }
    }
    for(int L = 1; L < 32; L <<= 1) myval += (tid >= L) * __shfl_up(myval, L) ;
    if (tid < 28) pack_start_padded[tid] = myval - mycount;
    if (tid == 26) {
      pack_start_padded[tid + 1] = myval;
      int nbulk = nparticles - myval;
      packsizes[0] = nbulk;
      if (nbulk > bulkcapacity) {
	failed = true;
	*failureflag = true;
      }
    }
  }

  __global__ void pack(int nparticles, int nfloat2s) {
    if (failed) return;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int slot = gid / 3;

    int tid = threadIdx.x;

    __shared__ int start[28];

    if (tid < 28) start[tid] = pack_start_padded[tid];
    __syncthreads();

    int key9 = 9 * (slot >= start[9]) + 9 * (slot >= start[18]);
    int key3 = 3 * (slot >= start[key9 + 3]) + 3 * (slot >= start[key9 + 6]);
    int key1 = (slot >= start[key9 + key3 + 1]) + (slot >= start[key9 + key3 + 2]);

    int idpack = key9 + key3 + key1;

    if (slot >= start[27]) return;

    int offset = slot - start[idpack];
    int pid = __ldg(pack_buffers[idpack].scattered_indices + offset);

    int c = gid % 3;
    int d = c + 3 * offset;
    pack_buffers[idpack].buffer[d] = tex1Dfetch(texAllParticlesFloat2, c + 3 * pid);
  }

  __global__ void subindex_remote(uint nparticles_padded,
				  uint nparticles, int *partials, float2 *dstbuf, uchar4 *subindices) {
    uint warpid = threadIdx.x >> 5;
    uint localbase = 32 * (warpid + 4 * blockIdx.x);
    if (localbase >= nparticles_padded)  return;
    uint key9 = 9 * (localbase >= unpack_start_padded[9]) + 9 * (localbase >= unpack_start_padded[18]);
    uint key3 = 3 * (localbase >= unpack_start_padded[key9 + 3]) + 3 * (localbase >= unpack_start_padded[key9 + 6]);
    uint key1 = (localbase >= unpack_start_padded[key9 + key3 + 1]) + (localbase >= unpack_start_padded[key9 + key3 + 2]);
    int code = key9 + key3 + key1;
    int unpackbase = localbase - unpack_start_padded[code];

    uint nunpack = min(32, unpack_start[code + 1] - unpack_start[code] - unpackbase);

    if (nunpack == 0) return;
    float2 data0, data1, data2;

    read_AOS6f(unpack_buffers[code].buffer + 3 * unpackbase, nunpack, data0, data1, data2);

    uint laneid = threadIdx.x & 0x1f;

    int xcid, ycid, zcid, subindex;
    if (laneid < nunpack) {
      data0.x += XSIZE_SUBDOMAIN * ((code + 1) % 3 - 1);
      data0.y += YSIZE_SUBDOMAIN * ((code / 3 + 1) % 3 - 1);
      data1.x += ZSIZE_SUBDOMAIN * ((code / 9 + 1) % 3 - 1);

      xcid = (int)floor((double)data0.x + XSIZE_SUBDOMAIN / 2);
      ycid = (int)floor((double)data0.y + YSIZE_SUBDOMAIN / 2);
      zcid = (int)floor((double)data1.x + ZSIZE_SUBDOMAIN / 2);

      int cid = xcid + XSIZE_SUBDOMAIN * (ycid + YSIZE_SUBDOMAIN * zcid);
      subindex = atomicAdd(partials + cid, 1);
    }

    uint dstbase = unpack_start[code] + unpackbase;

    write_AOS6f(dstbuf + 3 * dstbase, nunpack, data0, data1, data2);
    if (laneid < nunpack) subindices[dstbase + laneid] = make_uchar4(xcid, ycid, zcid, subindex);
  }

  __global__ void scatter_indices(bool remote, uchar4 * subindices, int nparticles,
				  int * starts, uint * scattered_indices, int nscattered) {
    uint pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= nparticles) return;
    uchar4 entry = subindices[pid];
    int subindex = entry.w;

    if (subindex != 255) {
      int cid = entry.x + XSIZE_SUBDOMAIN * (entry.y + YSIZE_SUBDOMAIN * entry.z);
      int base = __ldg(starts + cid);

      pid |= remote << 31;
      scattered_indices[base + subindex] = pid;
    }
  }

  __forceinline__ __device__ void xchg_aos2f(int srclane0, int srclane1, int start, float& s0, float& s1) {
    float t0 = __shfl(s0, srclane0);
    float t1 = __shfl(s1, srclane1);

    s0 = start == 0 ? t0 : t1;
    s1 = start == 0 ? t1 : t0;
    s1 = __shfl_xor(s1, 1);
  }

  __forceinline__ __device__ void xchg_aos4f(int srclane0, int srclane1, int start, float3& s0, float3& s1) {
    xchg_aos2f(srclane0, srclane1, start, s0.x, s1.x);
    xchg_aos2f(srclane0, srclane1, start, s0.y, s1.y);
    xchg_aos2f(srclane0, srclane1, start, s0.z, s1.z);
  }

  __global__ void gather_particles(uint * scattered_indices,
				   float2 *  remoteparticles, int nremoteparticles,
				   int noldparticles,
				   int nparticles,
				   float2 * dstbuf,
				   float4 * xyzouvwo,
				   ushort4 * xyzo_half) {
    int warpid = threadIdx.x >> 5;
    int tid = threadIdx.x & 0x1f;

    int base = 32 * (warpid + 4 * blockIdx.x);
    int pid = base + tid;

    bool valid = (pid < nparticles);

    uint spid;
    if (valid) spid = scattered_indices[pid];
    float2 data0, data1, data2;

    if (valid) {
      bool remote = (spid >> 31) & 1;
      spid &= ~(1 << 31);
      if (remote) {
	data0 = __ldg(remoteparticles + 0 + 3 * spid);
	data1 = __ldg(remoteparticles + 1 + 3 * spid);
	data2 = __ldg(remoteparticles + 2 + 3 * spid);
      } else {
	data0 = tex1Dfetch(texAllParticlesFloat2, 0 + 3 * spid);
	data1 = tex1Dfetch(texAllParticlesFloat2, 1 + 3 * spid);
	data2 = tex1Dfetch(texAllParticlesFloat2, 2 + 3 * spid);
      }
    }
    int nsrc = min(32, nparticles - base);

    {
      int srclane0 = (32 * ((tid) & 0x1) + tid) >> 1;
      int srclane1 = (32 * ((tid + 1) & 0x1) + tid) >> 1;
      int start = tid % 2;
      int destbase = 2 * base;

      float3 s0 = make_float3(data0.x, data0.y, data1.x);
      float3 s1 = make_float3(data1.y, data2.x, data2.y);

      xchg_aos4f(srclane0, srclane1, start, s0, s1);

      if (tid < 2 * nsrc)
	xyzouvwo[destbase + tid] = make_float4(s0.x, s0.y, s0.z, 0);

      if (tid + 32 < 2 * nsrc)
	xyzouvwo[destbase + tid + 32] = make_float4(s1.x, s1.y, s1.z, 0);
    }

    if (tid < nsrc) {
	xyzo_half[base + tid] = make_ushort4(
					     __float2half_rn(data0.x),
					     __float2half_rn(data0.y),
					     __float2half_rn(data1.x), 0);
    }
    write_AOS6f(dstbuf + 3 * base, nsrc, data0, data1, data2);
  }
}}
