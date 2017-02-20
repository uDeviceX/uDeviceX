#ifndef WARPSIZE
#define WARPSIZE 32
#endif

namespace RedistributeParticlesKernels {
  using namespace RedistPart;
    __constant__ PackBuffer pack_buffers[27];
    __constant__ UnpackBuffer unpack_buffers[27];
    __device__ int pack_count[27], pack_start_padded[28];
    __constant__ int unpack_start[28], unpack_start_padded[28];
    __device__ bool failed;

    int ntexparticles = 0;
    float2 * texparticledata;
    texture<float, cudaTextureType1D> texAllParticles;
    texture<float2, cudaTextureType1D> texAllParticlesFloat2;

#if !defined(__CUDA_ARCH__)
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

    __global__ void setup() {
        if (threadIdx.x == 0) failed = false;
        if (threadIdx.x < 27) pack_count[threadIdx.x] = 0;
    }

    __global__ void scatter_halo_indices_pack(int np) {
        int pid = threadIdx.x + blockDim.x * blockIdx.x;
        if (pid < np)
        {
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
    int pid = _ACCESS(pack_buffers[idpack].scattered_indices + offset);

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
	    int base = _ACCESS(starts + cid);

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
                data0 = _ACCESS(remoteparticles + 0 + 3 * spid);
                data1 = _ACCESS(remoteparticles + 1 + 3 * spid);
                data2 = _ACCESS(remoteparticles + 2 + 3 * spid);
            } else {
                if (spid >= noldparticles)
		  cuda_printf("ooops pid %d spid %d noldp%d\n", pid, spid, noldparticles);

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

        if (tid < nsrc)
        {
            xyzo_half[base + tid] = make_ushort4(
                    __float2half_rn(data0.x),
                    __float2half_rn(data0.y),
                    __float2half_rn(data1.x), 0);
        }

        write_AOS6f(dstbuf + 3 * base, nsrc, data0, data1, data2);
    }
#undef _ACCESS
}

namespace RedistPart {
int pack_size(int code) { return send_sizes[code]; }
float pinned_data(int code, int entry) {return pinnedhost_sendbufs[code][entry]; }

void _waitall(MPI_Request * reqs, int n) {
  MPI_Status statuses[n];
  MPI_CHECK( MPI_Waitall(n, reqs, statuses) );
}

void redist_part_init(MPI_Comm _cartcomm)  {
  failure = new PinnedHostBuffer<bool>(1);
  packsizes = new PinnedHostBuffer<int>(27);
  compressed_cellcounts = new SimpleDeviceBuffer<unsigned char>
    (XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN);
  remote_particles = new SimpleDeviceBuffer<Particle>;
  subindices_remote= new SimpleDeviceBuffer<uchar4>
    (1.5 * numberdensity * (XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN -
			    (XSIZE_SUBDOMAIN - 2) * (YSIZE_SUBDOMAIN - 2) * (ZSIZE_SUBDOMAIN - 2)));
  subindices = new SimpleDeviceBuffer<uchar4>
    (1.5 * numberdensity * XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN);
  scattered_indices = new SimpleDeviceBuffer<uint>;
  
  nactiveneighbors  = 26; firstcall = true;
  int dims[3], periods[3], coords[3];
  MPI_CHECK(MPI_Comm_dup(_cartcomm, &cartcomm_rdst) );
  MPI_CHECK( MPI_Comm_rank(cartcomm_rdst, &myrank) );
  MPI_CHECK( MPI_Cart_get(cartcomm_rdst, 3, dims, periods, coords) );

    for(int i = 0; i < 27; ++i) {
        int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };
        recv_tags[i] = (3 - d[0]) % 3 + 3 * ((3 - d[1]) % 3 + 3 * ((3 - d[2]) % 3));
        int coordsneighbor[3];
        for(int c = 0; c < 3; ++c) coordsneighbor[c] = coords[c] + d[c];
        MPI_CHECK( MPI_Cart_rank(cartcomm_rdst, coordsneighbor, neighbor_ranks + i) );

        int nhalodir[3] =  {
            d[0] != 0 ? 1 : XSIZE_SUBDOMAIN,
            d[1] != 0 ? 1 : YSIZE_SUBDOMAIN,
            d[2] != 0 ? 1 : ZSIZE_SUBDOMAIN
        };

        int nhalocells = nhalodir[0] * nhalodir[1] * nhalodir[2];
        int estimate = numberdensity * safety_factor * nhalocells;
        CC(cudaMalloc(&packbuffers[i].scattered_indices, sizeof(int) * estimate));

        if (i && estimate) {
            CC(cudaHostAlloc(&pinnedhost_sendbufs[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
            CC(cudaHostGetDevicePointer(&packbuffers[i].buffer, pinnedhost_sendbufs[i], 0));

            CC(cudaHostAlloc(&pinnedhost_recvbufs[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
            CC(cudaHostGetDevicePointer(&unpackbuffers[i].buffer, pinnedhost_recvbufs[i], 0));
        } else {
            CC(cudaMalloc(&packbuffers[i].buffer, sizeof(float) * 6 * estimate));
            unpackbuffers[i].buffer = packbuffers[i].buffer;
            pinnedhost_sendbufs[i] = NULL;
            pinnedhost_recvbufs[i] = NULL;
        }
        packbuffers[i].capacity = estimate;
        unpackbuffers[i].capacity = estimate;
        default_message_sizes[i] = estimate;
    }

    RedistributeParticlesKernels::texAllParticles.channelDesc = cudaCreateChannelDesc<float>();
    RedistributeParticlesKernels::texAllParticles.filterMode = cudaFilterModePoint;
    RedistributeParticlesKernels::texAllParticles.mipmapFilterMode = cudaFilterModePoint;
    RedistributeParticlesKernels::texAllParticles.normalized = 0;

    RedistributeParticlesKernels::texAllParticlesFloat2.channelDesc = cudaCreateChannelDesc<float2>();
    RedistributeParticlesKernels::texAllParticlesFloat2.filterMode = cudaFilterModePoint;
    RedistributeParticlesKernels::texAllParticlesFloat2.mipmapFilterMode = cudaFilterModePoint;
    RedistributeParticlesKernels::texAllParticlesFloat2.normalized = 0;

    CC(cudaEventCreate(&evpacking, cudaEventDisableTiming));
    CC(cudaEventCreate(&evsizes, cudaEventDisableTiming));

    CC(cudaFuncSetCacheConfig( RedistributeParticlesKernels::gather_particles, cudaFuncCachePreferL1 ) );
}

void _post_recv() {
    for(int i = 1, c = 0; i < 27; ++i)
        if (default_message_sizes[i])
            MPI_CHECK( MPI_Irecv(recv_sizes + i, 1, MPI_INTEGER, neighbor_ranks[i], basetag + recv_tags[i], cartcomm_rdst, recvcountreq + c++) );
        else
            recv_sizes[i] = 0;

    for(int i = 1, c = 0; i < 27; ++i)
        if (default_message_sizes[i])
            MPI_CHECK( MPI_Irecv(pinnedhost_recvbufs[i], default_message_sizes[i] * 6, MPI_FLOAT,
                        neighbor_ranks[i], basetag + recv_tags[i] + 333, cartcomm_rdst, recvmsgreq + c++) );
}

void _adjust_send_buffers(int requested_capacities[27]) {
  for(int i = 0; i < 27; ++i) {
    if (requested_capacities[i] <= packbuffers[i].capacity)
      continue;
    
    int capacity = requested_capacities[i];
    
    CC(cudaFree(packbuffers[i].scattered_indices));
    CC(cudaMalloc(&packbuffers[i].scattered_indices, sizeof(int) * capacity));
    
    if (i) {
      CC(cudaFreeHost(pinnedhost_sendbufs[i]));
      
      CC(cudaHostAlloc(&pinnedhost_sendbufs[i], sizeof(float) * 6 * capacity, cudaHostAllocMapped));
      CC(cudaHostGetDevicePointer(&packbuffers[i].buffer, pinnedhost_sendbufs[i], 0));
      
      packbuffers[i].capacity = capacity;
    }
    else {
      CC(cudaFree(packbuffers[i].buffer));
      
      CC(cudaMalloc(&packbuffers[i].buffer, sizeof(float) * 6 * capacity));
      unpackbuffers[i].buffer = packbuffers[i].buffer;
      
      packbuffers[i].capacity = capacity;
            unpackbuffers[i].capacity = capacity;
    }
  }
}

bool _adjust_recv_buffers(int requested_capacities[27]) {
    bool haschanged = false;
    for(int i = 0; i < 27; ++i) {
        if (requested_capacities[i] <= unpackbuffers[i].capacity) continue;
            
        haschanged = true;
        int capacity = requested_capacities[i];
        if (i) {
            //preserve-resize policy
            float * old = pinnedhost_recvbufs[i];

            CC(cudaHostAlloc(&pinnedhost_recvbufs[i], sizeof(float) * 6 * capacity, cudaHostAllocMapped));
            CC(cudaHostGetDevicePointer(&unpackbuffers[i].buffer, pinnedhost_recvbufs[i], 0));

            CC(cudaMemcpy(pinnedhost_recvbufs[i], old, sizeof(float) * 6 * unpackbuffers[i].capacity,
                        cudaMemcpyHostToHost));

            CC(cudaFreeHost(old));
        }
        else {
            printf("_adjust_recv_buffers i==0 ooooooooooooooops %d , req %d!!\n", unpackbuffers[i].capacity, capacity);
            abort();
        }

        unpackbuffers[i].capacity = capacity;
    }
    return haschanged;
}

void pack(Particle * particles, int nparticles, cudaStream_t mystream) {
    bool secondchance = false;
    if (firstcall) _post_recv();
    size_t textureoffset;
    if (nparticles)
        CC(cudaBindTexture(&textureoffset, &RedistributeParticlesKernels::texAllParticles, particles,
                    &RedistributeParticlesKernels::texAllParticles.channelDesc,
                    sizeof(float) * 6 * nparticles));

    if (nparticles)
        CC(cudaBindTexture(&textureoffset, &RedistributeParticlesKernels::texAllParticlesFloat2, particles,
                    &RedistributeParticlesKernels::texAllParticlesFloat2.channelDesc,
                    sizeof(float) * 6 * nparticles));

    RedistributeParticlesKernels::ntexparticles = nparticles;
    RedistributeParticlesKernels::texparticledata = (float2 *)particles;
pack_attempt:
    CC(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::pack_buffers, packbuffers,
                sizeof(PackBuffer) * 27, 0, cudaMemcpyHostToDevice, mystream));

    (*failure->data) = false;
    RedistributeParticlesKernels::setup<<<1, 32, 0, mystream>>>();

    if (nparticles)
        RedistributeParticlesKernels::scatter_halo_indices_pack<<< (nparticles + 127) / 128, 128, 0, mystream>>>(nparticles);

    RedistributeParticlesKernels::tiny_scan<<<1, 32, 0, mystream>>>(nparticles, packbuffers[0].capacity, packsizes->devptr, failure->devptr);

    CC(cudaEventRecord(evsizes, mystream));

    if (nparticles)
        RedistributeParticlesKernels::pack<<< (3 * nparticles + 127) / 128, 128, 0, mystream>>> (nparticles, nparticles * 3);

    CC(cudaEventRecord(evpacking, mystream));

    CC(cudaEventSynchronize(evsizes));

    if (*failure->data) {
        //wait for packing to finish
        CC(cudaEventSynchronize(evpacking));

        printf("pack RANK %d ...FAILED! Recovering now...\n", myrank);

        _adjust_send_buffers(packsizes->data);

        if (myrank == 0)
            for(int i = 0; i < 27; ++i)
                printf("ASD: %d\n", packsizes->data[i]);

        if (secondchance) {
            printf("...non siamo qui a far la ceretta allo yeti.\n");
            abort();
        }
        if (!secondchance) secondchance = true;
        goto pack_attempt;
    }
    CC(cudaPeekAtLastError());
}

void send() {
    if (!firstcall) _waitall(sendcountreq, nactiveneighbors);
    for(int i = 0; i < 27; ++i) send_sizes[i] = packsizes->data[i];
    nbulk = recv_sizes[0] = send_sizes[0];
    {
      int c = 0;
      for(int i = 1; i < 27; ++i)
	if (default_message_sizes[i])
	  MPI_CHECK( MPI_Isend(send_sizes + i, 1, MPI_INTEGER, neighbor_ranks[i], basetag + i, cartcomm_rdst, sendcountreq + c++) );
    }
    
    CC(cudaEventSynchronize(evpacking));
    
    if (!firstcall)
      _waitall(sendmsgreq, nsendmsgreq);
    
    nsendmsgreq = 0;
    for(int i = 1; i < 27; ++i)
      if (default_message_sizes[i]) {
            MPI_CHECK( MPI_Isend(pinnedhost_sendbufs[i], default_message_sizes[i] * 6, MPI_FLOAT, neighbor_ranks[i], basetag + i + 333,
                        cartcomm_rdst, sendmsgreq + nsendmsgreq) );
            ++nsendmsgreq;
      }
    
    for(int i = 1; i < 27; ++i)
      if (default_message_sizes[i] && send_sizes[i] > default_message_sizes[i]) {
	int count = send_sizes[i] - default_message_sizes[i];

	MPI_CHECK( MPI_Isend(pinnedhost_sendbufs[i] + default_message_sizes[i] * 6, count * 6, MPI_FLOAT,
			     neighbor_ranks[i], basetag + i + 666, cartcomm_rdst, sendmsgreq + nsendmsgreq) );
	++nsendmsgreq;
      }
}

void bulk(int nparticles, int * cellstarts, int * cellcounts, cudaStream_t mystream) {
    CC(cudaMemsetAsync(cellcounts, 0, sizeof(int) * XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN, mystream));

    subindices->resize(nparticles);

    if (nparticles)
        subindex_local<false><<< (nparticles + 127) / 128, 128, 0, mystream>>>
            (nparticles, RedistributeParticlesKernels::texparticledata, cellcounts, subindices->D);

    CC(cudaPeekAtLastError());
}

int recv_count(cudaStream_t mystream) {
    CC(cudaPeekAtLastError());

    _waitall(recvcountreq, nactiveneighbors);

    {
        static int usize[27], ustart[28], ustart_padded[28];

        usize[0] = 0;
        for(int i = 1; i < 27; ++i)
            usize[i] = recv_sizes[i] * (default_message_sizes[i] > 0);

        ustart[0] = 0;
        for(int i = 1; i < 28; ++i)
            ustart[i] = ustart[i - 1] + usize[i - 1];

        nexpected = nbulk + ustart[27];
        nhalo = ustart[27];

        ustart_padded[0] = 0;
        for(int i = 1; i < 28; ++i)
            ustart_padded[i] = ustart_padded[i - 1] + 32 * ((usize[i - 1] + 31) / 32);

        nhalo_padded = ustart_padded[27];

        CC(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::unpack_start, ustart,
                    sizeof(int) * 28, 0, cudaMemcpyHostToDevice, mystream));

        CC(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::unpack_start_padded, ustart_padded,
                    sizeof(int) * 28, 0, cudaMemcpyHostToDevice, mystream));
    }

    {
        remote_particles->resize(nhalo);
        subindices_remote->resize(nhalo);
        scattered_indices->resize(nexpected);
    }

    firstcall = false;
    return nexpected;
}

void recv_unpack(Particle * particles, float4 * xyzouvwo, ushort4 * xyzo_half, int nparticles,
					int * cellstarts, int * cellcounts, cudaStream_t mystream) {
    _waitall(recvmsgreq, nactiveneighbors);

    bool haschanged = true;
    _adjust_recv_buffers(recv_sizes);

    if (haschanged)
        CC(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::unpack_buffers, unpackbuffers,
                    sizeof(UnpackBuffer) * 27, 0, cudaMemcpyHostToDevice, mystream));

    for(int i = 1; i < 27; ++i)
      if (default_message_sizes[i] && recv_sizes[i] > default_message_sizes[i]) {
	int count = recv_sizes[i] - default_message_sizes[i];
	
	MPI_Status status;
	MPI_CHECK( MPI_Recv(pinnedhost_recvbufs[i] + default_message_sizes[i] * 6, count * 6, MPI_FLOAT,
			    neighbor_ranks[i], basetag + recv_tags[i] + 666, cartcomm_rdst, &status) );
      }
    CC(cudaPeekAtLastError());

#ifndef NDEBUG
    CC(cudaMemset(remote_particles->D, 0xff, sizeof(Particle) * remote_particles->S));
#endif

    if (nhalo)
        RedistributeParticlesKernels::subindex_remote<<< (nhalo_padded + 127) / 128, 128, 0, mystream >>>
            (nhalo_padded, nhalo, cellcounts, (float2 *)remote_particles->D, subindices_remote->D);

    if (compressed_cellcounts->S)
        compress_counts<<< (compressed_cellcounts->S + 127) / 128, 128, 0, mystream >>>
            (compressed_cellcounts->S, (int4 *)cellcounts, (uchar4 *)compressed_cellcounts->D);

    scan(compressed_cellcounts->D, compressed_cellcounts->S, mystream, (uint *)cellstarts);

#ifndef NDEBUG
    CC(cudaMemset(scattered_indices->D, 0xff, sizeof(int) * scattered_indices->S));
#endif

    if (subindices->S)
        RedistributeParticlesKernels::scatter_indices<<< (subindices->S + 127) / 128, 128, 0, mystream>>>
            (false, subindices->D, subindices->S, cellstarts, scattered_indices->D, scattered_indices->S);

    if (nhalo)
        RedistributeParticlesKernels::scatter_indices<<< (nhalo + 127) / 128, 128, 0, mystream>>>
            (true, subindices_remote->D, nhalo, cellstarts, scattered_indices->D, scattered_indices->S);

    if (nparticles)
        RedistributeParticlesKernels::gather_particles<<< (nparticles + 127) / 128, 128, 0, mystream>>>
            (scattered_indices->D, (float2 *)remote_particles->D, nhalo,
             RedistributeParticlesKernels::ntexparticles, nparticles, (float2 *)particles, xyzouvwo, xyzo_half);

    CC(cudaPeekAtLastError());

    _post_recv();

    CC(cudaPeekAtLastError());
}

void _cancel_recv() {
  if (!firstcall) {
    _waitall(sendcountreq, nactiveneighbors);
    _waitall(sendmsgreq, nsendmsgreq);
    
    for(int i = 0; i < nactiveneighbors; ++i)
      MPI_CHECK( MPI_Cancel(recvcountreq + i) );
    
    for(int i = 0; i < nactiveneighbors; ++i)
      MPI_CHECK( MPI_Cancel(recvmsgreq + i) );
    
    firstcall = true;
  }
}

void adjust_message_sizes(ExpectedMessageSizes sizes) {
    _cancel_recv();

    nactiveneighbors = 0;
    for (int i = 1; i < 27; ++i) {
        int d[3] = { (i + 1) % 3, (i / 3 + 1) % 3, (i / 9 + 1) % 3 };
        int entry = d[0] + 3 * (d[1] + 3 * d[2]);

        int estimate = (int)ceil(safety_factor * sizes.msgsizes[entry]);
        estimate = 32 * ((estimate + 31) / 32);

        default_message_sizes[i] = estimate;
        nactiveneighbors += (estimate > 0);
    }

    _adjust_send_buffers(default_message_sizes);
    _adjust_recv_buffers(default_message_sizes);
}

void redist_part_close() {
    CC(cudaEventDestroy(evpacking));
    CC(cudaEventDestroy(evsizes));

    _cancel_recv();

    for(int i = 0; i < 27; ++i) {
        CC(cudaFree(packbuffers[i].scattered_indices));
        if (i) CC(cudaFreeHost(packbuffers[i].buffer));
        else   CC(cudaFree(packbuffers[i].buffer));
    }

    delete failure;
    delete packsizes;
    delete compressed_cellcounts;
    delete remote_particles;
    delete subindices_remote;
    delete subindices;
    delete scattered_indices;
}
}
