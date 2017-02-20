/*
 *  solute-exchange.cu
 *  Part of uDeviceX/mpi-dpd/
 */

#include <dpd-rng.h>
#include <vector>
#include <cstdio>
#include <mpi.h>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"
#include "common-kernels.h"
#include "solute-exchange.h"
#include "fsi.h"
#include "contact.h"

extern ComputeFSI* fsi;
extern ComputeContact* contact;

namespace SolutePUP {
  __constant__ int ccapacities[26], *scattered_indices[26];
}

SoluteExchange::SoluteExchange(MPI_Comm _cartcomm) {
  iterationcount = -1;
  packstotalstart = new SimpleDeviceBuffer<int>(27);
  host_packstotalstart = new PinnedHostBuffer<int>(27);
  host_packstotalcount = new PinnedHostBuffer<int>(26);

  packscount = new SimpleDeviceBuffer<int>;
  packsstart = new SimpleDeviceBuffer<int>;
  packsoffset = new SimpleDeviceBuffer<int>;

  packbuf = new SimpleDeviceBuffer<Particle>;

  MPI_CHECK(MPI_Comm_dup(_cartcomm, &cartcomm));
  MPI_CHECK(MPI_Comm_size(cartcomm, &nranks));
  MPI_CHECK(MPI_Cart_get(cartcomm, 3, dims, periods, coords));
  MPI_CHECK(MPI_Comm_rank(cartcomm, &myrank));

  for (int i = 0; i < 26; ++i) {
    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

    recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

    int coordsneighbor[3];
    for (int c = 0; c < 3; ++c) coordsneighbor[c] = coords[c] + d[c];

    MPI_CHECK(MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i));

    int estimate = 1;
    remote[i].preserve_resize(estimate);
    local[i].resize(estimate);
    local[i].update();

    CC(cudaMemcpyToSymbol(SolutePUP::ccapacities,
			  &local[i].scattered_indices.capacity, sizeof(int),
			  sizeof(int) * i, cudaMemcpyHostToDevice));
    CC(cudaMemcpyToSymbol(SolutePUP::scattered_indices,
			  &local[i].scattered_indices.D, sizeof(int *),
			  sizeof(int *) * i, cudaMemcpyHostToDevice));
  }

  _adjust_packbuffers();

  CC(cudaEventCreateWithFlags(&evPpacked,
			      cudaEventDisableTiming | cudaEventBlockingSync));
  CC(cudaEventCreateWithFlags(&evAcomputed,
			      cudaEventDisableTiming | cudaEventBlockingSync));

  CC(cudaPeekAtLastError());
}

namespace SolutePUP {
__device__ bool failed;

__global__ void init() { failed = false; }

__constant__ int coffsets[26];

__global__ void scatter_indices(float2 *particles,
				int nparticles, int *counts) {
  int warpid = threadIdx.x >> 5;
  int base = 32 * (warpid + 4 * blockIdx.x);
  int nsrc = min(32, nparticles - base);

  float2 s0, s1, s2;
  read_AOS6f(particles + 3 * base, nsrc, s0, s1, s2);

  int lane = threadIdx.x & 0x1f;
  int pid = base + lane;

  if (lane >= nsrc) return;

  enum {
    HXSIZE = XSIZE_SUBDOMAIN / 2,
    HYSIZE = YSIZE_SUBDOMAIN / 2,
    HZSIZE = ZSIZE_SUBDOMAIN / 2
  };

  int halocode[3] = {
      -1 + (int)(s0.x >= -HXSIZE + 1) + (int)(s0.x >= HXSIZE - 1),
      -1 + (int)(s0.y >= -HYSIZE + 1) + (int)(s0.y >= HYSIZE - 1),
      -1 + (int)(s1.x >= -HZSIZE + 1) + (int)(s1.x >= HZSIZE - 1)};

  if (halocode[0] == 0 && halocode[1] == 0 && halocode[2] == 0) return;

// faces
#pragma unroll 3
  for (int d = 0; d < 3; ++d)
    if (halocode[d]) {
      int xterm = (halocode[0] * (d == 0) + 2) % 3;
      int yterm = (halocode[1] * (d == 1) + 2) % 3;
      int zterm = (halocode[2] * (d == 2) + 2) % 3;

      int bagid = xterm + 3 * (yterm + 3 * zterm);
      int myid = coffsets[bagid] + atomicAdd(counts + bagid, 1);

      if (myid < ccapacities[bagid]) scattered_indices[bagid][myid] = pid;
    }

// edges
#pragma unroll 3
  for (int d = 0; d < 3; ++d)
    if (halocode[(d + 1) % 3] && halocode[(d + 2) % 3]) {
      int xterm = (halocode[0] * (d != 0) + 2) % 3;
      int yterm = (halocode[1] * (d != 1) + 2) % 3;
      int zterm = (halocode[2] * (d != 2) + 2) % 3;

      int bagid = xterm + 3 * (yterm + 3 * zterm);
      int myid = coffsets[bagid] + atomicAdd(counts + bagid, 1);

      if (myid < ccapacities[bagid]) scattered_indices[bagid][myid] = pid;
    }

  // one corner
  if (halocode[0] && halocode[1] && halocode[2]) {
    int xterm = (halocode[0] + 2) % 3;
    int yterm = (halocode[1] + 2) % 3;
    int zterm = (halocode[2] + 2) % 3;

    int bagid = xterm + 3 * (yterm + 3 * zterm);
    int myid = coffsets[bagid] + atomicAdd(counts + bagid, 1);

    if (myid < ccapacities[bagid]) scattered_indices[bagid][myid] = pid;
  }
}

__global__ void tiny_scan(int *counts,
			  int *oldtotalcounts,
			  int *totalcounts, int *paddedstarts) {
  int tid = threadIdx.x;

  int mycount = 0;

  if (tid < 26) {
    mycount = counts[tid];

    if (mycount > ccapacities[tid]) failed = true;

    if (totalcounts && oldtotalcounts) {
      int newcount = mycount + oldtotalcounts[tid];
      totalcounts[tid] = newcount;

      if (newcount > ccapacities[tid]) failed = true;
    }
  }

  if (paddedstarts) {
    int myscan = mycount = 32 * ((mycount + 31) / 32);

    for (int L = 1; L < 32; L <<= 1)
      myscan += (tid >= L) * __shfl_up(myscan, L);

    if (tid < 27) paddedstarts[tid] = myscan - mycount;
  }
}

__constant__ int ccounts[26], cbases[27], cpaddedstarts[27];

__global__ void pack(float2 *particles, int nparticles,
		     float2 *buffer, int nbuffer,
		     int soluteid) {
#if !defined(__CUDA_ARCH__)
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

  if (failed) return;

  int warpid = threadIdx.x >> 5;
  int npack_padded = cpaddedstarts[26];

  for (int localbase = 32 * (warpid + 4 * blockIdx.x); localbase < npack_padded;
       localbase += gridDim.x * blockDim.x) {
    int key9 = 9 * ((int)(localbase >= cpaddedstarts[9]) +
			  (int)(localbase >= cpaddedstarts[18]));

    int key3 = 3 * ((int)(localbase >= cpaddedstarts[key9 + 3]) +
			  (int)(localbase >= cpaddedstarts[key9 + 6]));

    int key1 = (int)(localbase >= cpaddedstarts[key9 + key3 + 1]) +
		     (int)(localbase >= cpaddedstarts[key9 + key3 + 2]);

    int code = key9 + key3 + key1;
    int packbase = localbase - cpaddedstarts[code];

    int npack = min(32, ccounts[code] - packbase);

    int lane = threadIdx.x & 0x1f;

    float2 s0, s1, s2;

    if (lane < npack) {
      int entry = coffsets[code] + packbase + lane;
      int pid = _ACCESS(scattered_indices[code] + entry);

      int entry2 = 3 * pid;

      s0 = _ACCESS(particles + entry2);
      s1 = _ACCESS(particles + entry2 + 1);
      s2 = _ACCESS(particles + entry2 + 2);

      s0.x -= ((code + 2) % 3 - 1) * XSIZE_SUBDOMAIN;
      s0.y -= ((code / 3 + 2) % 3 - 1) * YSIZE_SUBDOMAIN;
      s1.x -= ((code / 9 + 2) % 3 - 1) * ZSIZE_SUBDOMAIN;
    }
    write_AOS6f(buffer + 3 * (cbases[code] + coffsets[code] + packbase), npack,
		s0, s1, s2);
  }
}
}

void SoluteExchange::_pack_attempt(cudaStream_t stream) {
#ifndef NDEBUG
  CC(cudaMemsetAsync(packbuf->D, 0xff, sizeof(Particle) * packbuf->capacity,
		     stream));
  memset(host_packbuf.data, 0xff, sizeof(Particle) * packbuf->capacity);

  for (int i = 0; i < 26; ++i) {
    CC(cudaMemsetAsync(local[i].scattered_indices.D, 0x8f,
		       sizeof(int) * local[i].scattered_indices.capacity,
		       stream));
    CC(cudaMemsetAsync(local[i].result.data, 0xff,
		       sizeof(Acceleration) * local[i].result.capacity,
		       stream));
  }
#endif
  CC(cudaPeekAtLastError());

  if (packscount->S)
    CC(cudaMemsetAsync(packscount->D, 0, sizeof(int) * packscount->S, stream));

  if (packsoffset->S)
    CC(cudaMemsetAsync(packsoffset->D, 0, sizeof(int) * packsoffset->S, stream));

  if (packsstart->S)
    CC(cudaMemsetAsync(packsstart->D, 0, sizeof(int) * packsstart->S, stream));

  SolutePUP::init<<<1, 1, 0, stream>>>();

  for (int i = 0; i < wsolutes.size(); ++i) {
    ParticlesWrap it = wsolutes[i];

    if (it.n) {
      CC(cudaMemcpyToSymbolAsync(SolutePUP::coffsets, packsoffset->D + 26 * i,
				 sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice,
				 stream));

      SolutePUP::scatter_indices<<<(it.n + 127) / 128, 128, 0, stream>>>(
	  (float2 *)it.p, it.n, packscount->D + i * 26);
    }

    SolutePUP::tiny_scan<<<1, 32, 0, stream>>>(
	packscount->D + i * 26, packsoffset->D + 26 * i,
	packsoffset->D + 26 * (i + 1), packsstart->D + i * 27);

    CC(cudaPeekAtLastError());
  }

  CC(cudaMemcpyAsync(host_packstotalcount->data,
		     packsoffset->D + 26 * wsolutes.size(), sizeof(int) * 26,
		     cudaMemcpyDeviceToHost, stream));

  SolutePUP::tiny_scan<<<1, 32, 0, stream>>>(
      packsoffset->D + 26 * wsolutes.size(), NULL, NULL, packstotalstart->D);

  CC(cudaMemcpyAsync(host_packstotalstart->data, packstotalstart->D,
		     sizeof(int) * 27, cudaMemcpyDeviceToHost, stream));

  CC(cudaMemcpyToSymbolAsync(SolutePUP::cbases, packstotalstart->D,
			     sizeof(int) * 27, 0, cudaMemcpyDeviceToDevice,
			     stream));

  for (int i = 0; i < wsolutes.size(); ++i) {
    ParticlesWrap it = wsolutes[i];

    if (it.n) {
      CC(cudaMemcpyToSymbolAsync(SolutePUP::coffsets, packsoffset->D + 26 * i,
				 sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice,
				 stream));
      CC(cudaMemcpyToSymbolAsync(SolutePUP::ccounts, packscount->D + 26 * i,
				 sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice,
				 stream));
      CC(cudaMemcpyToSymbolAsync(SolutePUP::cpaddedstarts,
				 packsstart->D + 27 * i, sizeof(int) * 27, 0,
				 cudaMemcpyDeviceToDevice, stream));

      SolutePUP::pack<<<14 * 16, 128, 0, stream>>>(
	  (float2 *)it.p, it.n, (float2 *)packbuf->D, packbuf->capacity, i);
    }
  }

  CC(cudaEventRecord(evPpacked, stream));

  CC(cudaPeekAtLastError());
}

void SoluteExchange::pack_p(cudaStream_t stream) {
  if (wsolutes.size() == 0) return;

  ++iterationcount;

  packscount->resize(26 * wsolutes.size());
  packsoffset->resize(26 * (wsolutes.size() + 1));
  packsstart->resize(27 * wsolutes.size());

  _pack_attempt(stream);
}

void SoluteExchange::post_p(cudaStream_t stream, cudaStream_t downloadstream) {
  if (wsolutes.size() == 0) return;

  CC(cudaPeekAtLastError());

  // consolidate the packing
  {
    CC(cudaEventSynchronize(evPpacked));

    if (iterationcount == 0)
      _postrecvC();
    else
      _wait(reqsendC);

    for (int i = 0; i < 26; ++i) send_counts[i] = host_packstotalcount->data[i];

    bool packingfailed = false;

    for (int i = 0; i < 26; ++i)
      packingfailed |= send_counts[i] > local[i].capacity();

    if (packingfailed) {
      for (int i = 0; i < 26; ++i) local[i].resize(send_counts[i]);

      int newcapacities[26];
      for (int i = 0; i < 26; ++i) newcapacities[i] = local[i].capacity();

      CC(cudaMemcpyToSymbolAsync(SolutePUP::ccapacities, newcapacities,
				 sizeof(newcapacities), 0,
				 cudaMemcpyHostToDevice, stream));

      int *newindices[26];
      for (int i = 0; i < 26; ++i) newindices[i] = local[i].scattered_indices.D;

      CC(cudaMemcpyToSymbolAsync(SolutePUP::scattered_indices, newindices,
				 sizeof(newindices), 0, cudaMemcpyHostToDevice,
				 stream));

      _adjust_packbuffers();

      _pack_attempt(stream);

      CC(cudaStreamSynchronize(stream));
    }

    for (int i = 0; i < 26; ++i) local[i].resize(send_counts[i]);

    _postrecvA();

    if (iterationcount == 0) {
#ifndef _DUMBCRAY_
      _postrecvP();
#endif
    } else
      _wait(reqsendP);

    if (host_packstotalstart->data[26]) {
      CC(cudaMemcpyAsync(host_packbuf.data, packbuf->D,
			 sizeof(Particle) * host_packstotalstart->data[26],
			 cudaMemcpyDeviceToHost, downloadstream));
    }

    CC(cudaStreamSynchronize(downloadstream));
  }

  // post the sending of the packs
  {
    reqsendC.resize(26);

    for (int i = 0; i < 26; ++i)
      MPI_CHECK(MPI_Isend(send_counts + i, 1, MPI_INTEGER, dstranks[i],
			  TAGBASE_C + i, cartcomm, &reqsendC[i]));

    for (int i = 0; i < 26; ++i) {
      int start = host_packstotalstart->data[i];
      int count = send_counts[i];
      int expected = local[i].expected();

      MPI_Request reqP;

      _not_nan((float *)(host_packbuf.data + start), count * 6);

#ifdef _DUMBCRAY_
      MPI_CHECK(MPI_Isend(host_packbuf.data + start, count * 6, MPI_FLOAT,
			  dstranks[i], TAGBASE_P + i, cartcomm, &reqP));
#else
      MPI_CHECK(MPI_Isend(host_packbuf.data + start, expected * 6, MPI_FLOAT,
			  dstranks[i], TAGBASE_P + i, cartcomm, &reqP));
#endif

      reqsendP.push_back(reqP);

#ifndef _DUMBCRAY_
      if (count > expected) {
	MPI_Request reqP2;
	MPI_CHECK(MPI_Isend(host_packbuf.data + start + expected,
			    (count - expected) * 6, MPI_FLOAT, dstranks[i],
			    TAGBASE_P2 + i, cartcomm, &reqP2));

	reqsendP.push_back(reqP2);
      }
#endif
    }
  }
}

void SoluteExchange::recv_p(cudaStream_t uploadstream) {
  if (wsolutes.size() == 0) return;

  _wait(reqrecvC);
  _wait(reqrecvP);

  for (int i = 0; i < 26; ++i) {
    int count = recv_counts[i];
    int expected = remote[i].expected();

    remote[i].pmessage.resize(max(1, count));
    remote[i].preserve_resize(count);

#ifndef NDEBUG
    CC(cudaMemsetAsync(remote[i].dstate.D, 0xff,
		       sizeof(Particle) * remote[i].dstate.capacity,
		       uploadstream));
    CC(cudaMemsetAsync(remote[i].result.data, 0xff,
		       sizeof(Acceleration) * remote[i].result.capacity,
		       uploadstream));
#endif

    MPI_Status status;

#ifdef _DUMBCRAY_
    MPI_CHECK(MPI_Recv(remote[i].hstate.data, count * 6, MPI_FLOAT, dstranks[i],
		       TAGBASE_P + recv_tags[i], cartcomm, &status));
#else
    if (count > expected)
      MPI_CHECK(MPI_Recv(&remote[i].pmessage.front() + expected,
			 (count - expected) * 6, MPI_FLOAT, dstranks[i],
			 TAGBASE_P2 + recv_tags[i], cartcomm, &status));

    memcpy(remote[i].hstate.data, &remote[i].pmessage.front(),
	   sizeof(Particle) * count);
#endif

    _not_nan((float *)remote[i].hstate.data, count * 6);
  }

  _postrecvC();

  for (int i = 0; i < 26; ++i)
    CC(cudaMemcpyAsync(remote[i].dstate.D, remote[i].hstate.data,
		       sizeof(Particle) * remote[i].hstate.size,
		       cudaMemcpyHostToDevice, uploadstream));
}

void SoluteExchange::halo(cudaStream_t uploadstream, cudaStream_t stream) {
  if (wsolutes.size() == 0) return;

  if (iterationcount) _wait(reqsendA);

  ParticlesWrap halos[26];

  for (int i = 0; i < 26; ++i)
    halos[i] = ParticlesWrap(remote[i].dstate.D, remote[i].dstate.S,
			     remote[i].result.devptr);

  CC(cudaStreamSynchronize(uploadstream));

  /** here was visitor  **/
  fsi->halo(halos, stream);
  if (contactforces) contact->halo(halos, stream);
  /***********************/

  CC(cudaPeekAtLastError());

  CC(cudaEventRecord(evAcomputed, stream));

  for (int i = 0; i < 26; ++i) local[i].update();

#ifndef _DUMBCRAY_
  _postrecvP();
#endif
}

void SoluteExchange::post_a() {
  if (wsolutes.size() == 0) return;

  CC(cudaEventSynchronize(evAcomputed));

  reqsendA.resize(26);
  for (int i = 0; i < 26; ++i)
    MPI_CHECK(MPI_Isend(remote[i].result.data, remote[i].result.size * 3,
			MPI_FLOAT, dstranks[i], TAGBASE_A + i, cartcomm,
			&reqsendA[i]));
}

namespace SolutePUP {
__constant__ float *recvbags[26];

__global__ void unpack(float *accelerations, int nparticles) {
  int npack_padded = cpaddedstarts[26];

  for (int gid = threadIdx.x + blockDim.x * blockIdx.x; gid < 3 * npack_padded;
       gid += blockDim.x * gridDim.x) {
    int pid = gid / 3;

    if (pid >= npack_padded) return;

    int key9 =
	9 * ((int)(pid >= cpaddedstarts[9]) + (int)(pid >= cpaddedstarts[18]));

    int key3 = 3 * ((int)(pid >= cpaddedstarts[key9 + 3]) +
			  (int)(pid >= cpaddedstarts[key9 + 6]));

    int key1 = (int)(pid >= cpaddedstarts[key9 + key3 + 1]) +
		     (int)(pid >= cpaddedstarts[key9 + key3 + 2]);

    int code = key9 + key3 + key1;
    int lpid = pid - cpaddedstarts[code];

    if (lpid >= ccounts[code]) continue;

    int component = gid % 3;

    int entry = coffsets[code] + lpid;
    float myval = _ACCESS(recvbags[code] + component + 3 * entry);
    int dpid = _ACCESS(scattered_indices[code] + entry);

    atomicAdd(accelerations + 3 * dpid + component, myval);
  }
}
}

void SoluteExchange::recv_a(cudaStream_t stream) {
  CC(cudaPeekAtLastError());

  if (wsolutes.size() == 0) return;

  {
    float *recvbags[26];

    for (int i = 0; i < 26; ++i) recvbags[i] = (float *)local[i].result.devptr;

    CC(cudaMemcpyToSymbolAsync(SolutePUP::recvbags, recvbags, sizeof(recvbags),
			       0, cudaMemcpyHostToDevice, stream));
  }

  _wait(reqrecvA);

  for (int i = 0; i < wsolutes.size(); ++i) {
    ParticlesWrap it = wsolutes[i];

    if (it.n) {
      CC(cudaMemcpyToSymbolAsync(SolutePUP::cpaddedstarts,
				 packsstart->D + 27 * i, sizeof(int) * 27, 0,
				 cudaMemcpyDeviceToDevice, stream));
      CC(cudaMemcpyToSymbolAsync(SolutePUP::ccounts, packscount->D + 26 * i,
				 sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice,
				 stream));
      CC(cudaMemcpyToSymbolAsync(SolutePUP::coffsets, packsoffset->D + 26 * i,
				 sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice,
				 stream));

      SolutePUP::unpack<<<16 * 14, 128, 0, stream>>>((float *)it.a, it.n);
    }
    CC(cudaPeekAtLastError());
  }
}

SoluteExchange::~SoluteExchange() {
  MPI_CHECK(MPI_Comm_free(&cartcomm));

  CC(cudaEventDestroy(evPpacked));
  CC(cudaEventDestroy(evAcomputed));

  delete packstotalstart;
  delete host_packstotalstart;
  delete host_packstotalcount;

  delete packscount;
  delete packsstart;
  delete packsoffset;
  delete packbuf;
}
