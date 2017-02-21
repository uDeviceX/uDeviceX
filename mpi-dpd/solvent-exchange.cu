#include <mpi.h>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"
#include "solvent-exchange.h"

SolventExchange::SolventExchange(MPI_Comm _cartcomm, int basetag)
  : basetag(basetag), firstpost(true), nactive(26) {
  safety_factor =
    getenv("HEX_COMM_FACTOR") ? atof(getenv("HEX_COMM_FACTOR")) : 1.2;

  MC(MPI_Comm_dup(_cartcomm, &cartcomm));
  MC(MPI_Comm_rank(cartcomm, &myrank));
  MC(MPI_Comm_size(cartcomm, &nranks));
  MC(MPI_Cart_get(cartcomm, 3, dims, periods, coords));

  for (int i = 0; i < 26; ++i) {
    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
    recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
    int coordsneighbor[3];
    for (int c = 0; c < 3; ++c) coordsneighbor[c] = coords[c] + d[c];
    MC(MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i));
    halosize[i].x = d[0] != 0 ? 1 : XSIZE_SUBDOMAIN;
    halosize[i].y = d[1] != 0 ? 1 : YSIZE_SUBDOMAIN;
    halosize[i].z = d[2] != 0 ? 1 : ZSIZE_SUBDOMAIN;

    int nhalocells = halosize[i].x * halosize[i].y * halosize[i].z;

    int estimate = numberdensity * safety_factor * nhalocells;
    estimate = 32 * ((estimate + 31) / 32);

    recvhalos[i].setup(estimate, nhalocells);
    sendhalos[i].setup(estimate, nhalocells);
  }

  CC(cudaHostAlloc((void **)&required_send_bag_size_host, sizeof(int) * 26,
                   cudaHostAllocMapped));
  CC(cudaHostGetDevicePointer(&required_send_bag_size,
                              required_send_bag_size_host, 0));
  CC(cudaEventCreateWithFlags(&evfillall, cudaEventDisableTiming));
  CC(cudaEventCreateWithFlags(&evdownloaded,
                              cudaEventDisableTiming | cudaEventBlockingSync));
}

namespace PackingHalo {
  int ncells;

  __constant__ int cellpackstarts[27];

  struct CellPackSOA {
    int *start, *count, *scan, size;
    bool enabled;
  };

  __constant__ CellPackSOA cellpacks[26];

  __global__ void count_all(int *cellsstart,
			    int *cellscount, int ntotalcells) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid >= cellpackstarts[26]) return;

    int key9 =
      9 * ((gid >= cellpackstarts[9]) + (gid >= cellpackstarts[18]));
    int key3 = 3 * ((gid >= cellpackstarts[key9 + 3]) +
		    (gid >= cellpackstarts[key9 + 6]));
    int key1 = (gid >= cellpackstarts[key9 + key3 + 1]) +
      (gid >= cellpackstarts[key9 + key3 + 2]);
    int code = key9 + key3 + key1;
    int d[3] = {(code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1,
		(code / 9 + 2) % 3 - 1};
    int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};

    int halo_start[3];
    for (int c = 0; c < 3; ++c)
      halo_start[c] = max(d[c] * L[c] - L[c] / 2 - 1, -L[c] / 2);

    int halo_size[3];
    for (int c = 0; c < 3; ++c)
      halo_size[c] = min(d[c] * L[c] + L[c] / 2 + 1, L[c] / 2) - halo_start[c];

    int ndstcells = halo_size[0] * halo_size[1] * halo_size[2];
    int dstcid = gid - cellpackstarts[code];

    if (dstcid < ndstcells) {
      int dstcellpos[3] = {dstcid % halo_size[0],
			   (dstcid / halo_size[0]) % halo_size[1],
			   dstcid / (halo_size[0] * halo_size[1])};

      int srccellpos[3];
      for (int c = 0; c < 3; ++c)
	srccellpos[c] = halo_start[c] + dstcellpos[c] + L[c] / 2;

      int srcentry =
        srccellpos[0] +
        XSIZE_SUBDOMAIN * (srccellpos[1] + YSIZE_SUBDOMAIN * srccellpos[2]);
      int enabled = cellpacks[code].enabled;

      cellpacks[code].start[dstcid] = enabled * cellsstart[srcentry];
      cellpacks[code].count[dstcid] = enabled * cellscount[srcentry];
    } else if (dstcid == ndstcells) {
      cellpacks[code].start[dstcid] = 0;
      cellpacks[code].count[dstcid] = 0;
    }
  }

  __constant__ int *srccells[26 * 2], *dstcells[26 * 2];
  template <int slot> __global__ void copycells(int n) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid >= cellpackstarts[26]) return;

    int key9 =
      9 * ((gid >= cellpackstarts[9]) + (gid >= cellpackstarts[18]));
    int key3 = 3 * ((gid >= cellpackstarts[key9 + 3]) +
		    (gid >= cellpackstarts[key9 + 6]));
    int key1 = (gid >= cellpackstarts[key9 + key3 + 1]) +
      (gid >= cellpackstarts[key9 + key3 + 2]);

    int idpack = key9 + key3 + key1;

    int offset = gid - cellpackstarts[idpack];

    dstcells[idpack + 26 * slot][offset] = srccells[idpack + 26 * slot][offset];
  }

  template <int NWARPS> __global__ void scan_diego() {
    __shared__ int shdata[32];

    int code = blockIdx.x;
    int *count = cellpacks[code].count;
    int *start = cellpacks[code].scan;
    int n = cellpacks[code].size;

    int tid = threadIdx.x;
    int laneid = threadIdx.x & 0x1f;
    int warpid = threadIdx.x >> 5;

    int lastval = 0;
    for (int sourcebase = 0; sourcebase < n; sourcebase += 32 * NWARPS) {
      int sourceid = sourcebase + tid;
      int mycount = 0, myscan = 0;
      if (sourceid < n) myscan = mycount = count[sourceid];
      if (tid == 0) myscan += lastval;

      for (int L = 1; L < 32; L <<= 1) {
	int val = __shfl_up(myscan, L);
	if (laneid >= L) myscan += val;
      }

      if (laneid == 31) shdata[warpid] = myscan;
      __syncthreads();
      if (warpid == 0) {
	int gs = 0;
	if (laneid < NWARPS) gs = shdata[tid];
	for (int L = 1; L < 32; L <<= 1) {
	  int val = __shfl_up(gs, L);
	  if (laneid >= L) gs += val;
	}

	shdata[tid] = gs;
	lastval = __shfl(gs, 31);
      }
      __syncthreads();
      if (warpid) myscan += shdata[warpid - 1];
      __syncthreads();
      if (sourceid < n) start[sourceid] = myscan - mycount;
    }
  }

  struct SendBagInfo {
    int *start_src, *count_src, *start_dst;
    int bagsize, *scattered_entries;
    Particle *dbag, *hbag;
  };

  __constant__ SendBagInfo baginfos[26];

  __global__ void fill_all(Particle *particles, int np,
			   int *required_bag_size) {
    int gcid = (threadIdx.x >> 4) + 2 * blockIdx.x;
    if (gcid >= cellpackstarts[26]) return;
    int key9 =
			  9 * ((gcid >= cellpackstarts[9]) + (gcid >= cellpackstarts[18]));
    int key3 = 3 * ((gcid >= cellpackstarts[key9 + 3]) +
		    (gcid >= cellpackstarts[key9 + 6]));
    int key1 = (gcid >= cellpackstarts[key9 + key3 + 1]) +
      (gcid >= cellpackstarts[key9 + key3 + 2]);
    int code = key9 + key3 + key1;
    int cellid = gcid - cellpackstarts[code];
    int tid = threadIdx.x & 0xf;
    int base_src = baginfos[code].start_src[cellid];
    int base_dst = baginfos[code].start_dst[cellid];
    int nsrc =
      min(baginfos[code].count_src[cellid], baginfos[code].bagsize - base_dst);
    int nfloats = nsrc * 6;
    for (int i = 2 * tid; i < nfloats; i += warpSize) {
      int lpid = i / 6;
      int dpid = base_dst + lpid;
      int spid = base_src + lpid;
      int c = i % 6;
      float2 word = *(float2 *)&particles[spid].x[c];
      *(float2 *)&baginfos[code].dbag[dpid].x[c] = word;
    }
    for (int lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
      int dpid = base_dst + lpid;
      int spid = base_src + lpid;
      baginfos[code].scattered_entries[dpid] = spid;
    }
    if (gcid + 1 == cellpackstarts[code + 1]) required_bag_size[code] = base_dst;
  }
}

void SolventExchange::_pack_all(Particle *p, int n,
                                bool update_baginfos,
                                cudaStream_t stream) {
  if (update_baginfos) {
    static PackingHalo::SendBagInfo baginfos[26];
    for (int i = 0; i < 26; ++i) {
      baginfos[i].start_src = sendhalos[i].tmpstart.D;
      baginfos[i].count_src = sendhalos[i].tmpcount.D;
      baginfos[i].start_dst = sendhalos[i].dcellstarts.D;
      baginfos[i].bagsize = sendhalos[i].dbuf.capacity;
      baginfos[i].scattered_entries = sendhalos[i].scattered_entries.D;
      baginfos[i].dbag = sendhalos[i].dbuf.D;
      baginfos[i].hbag = sendhalos[i].hbuf.data;
    }
    CC(cudaMemcpyToSymbolAsync(PackingHalo::baginfos, baginfos,
                               sizeof(baginfos), 0, cudaMemcpyHostToDevice,
                               stream)); // peh: added stream
  }

  if (PackingHalo::ncells)
    PackingHalo::fill_all<<<(PackingHalo::ncells + 1) / 2, 32, 0, stream>>>(
									    p, n, required_send_bag_size);
  CC(cudaEventRecord(evfillall, stream));
}

void SolventExchange::pack(Particle *p, int n,
                           int *cellsstart,
                           int *cellscount, cudaStream_t stream) {
  CC(cudaPeekAtLastError());
  nlocal = n;
  if (firstpost) {
    {
      static int cellpackstarts[27];
      cellpackstarts[0] = 0;
      for (int i = 0, s = 0; i < 26; ++i)
        cellpackstarts[i + 1] =
	  (s += sendhalos[i].dcellstarts.S * (sendhalos[i].expected > 0));
      PackingHalo::ncells = cellpackstarts[26];
      CC(cudaMemcpyToSymbol(PackingHalo::cellpackstarts, cellpackstarts,
                            sizeof(cellpackstarts), 0, cudaMemcpyHostToDevice));
    }

    {
      static PackingHalo::CellPackSOA cellpacks[26];
      for (int i = 0; i < 26; ++i) {
        cellpacks[i].start = sendhalos[i].tmpstart.D;
        cellpacks[i].count = sendhalos[i].tmpcount.D;
        cellpacks[i].enabled = sendhalos[i].expected > 0;
        cellpacks[i].scan = sendhalos[i].dcellstarts.D;
        cellpacks[i].size = sendhalos[i].dcellstarts.S;
      }
      CC(cudaMemcpyToSymbol(PackingHalo::cellpacks, cellpacks,
                            sizeof(cellpacks), 0, cudaMemcpyHostToDevice));
    }
  }

  if (PackingHalo::ncells)
    PackingHalo::
      count_all<<<(PackingHalo::ncells + 127) / 128, 128, 0, stream>>>(
								       cellsstart, cellscount, PackingHalo::ncells);

  PackingHalo::scan_diego<32><<<26, 32 * 32, 0, stream>>>();
  CC(cudaPeekAtLastError());
  if (firstpost) post_expected_recv();
  else {
    MPI_Status statuses[26 * 2];
    MC(MPI_Waitall(nactive, sendcellsreq, statuses));
    MC(MPI_Waitall(nsendreq, sendreq, statuses));
    MC(MPI_Waitall(nactive, sendcountreq, statuses));
  }

  if (firstpost) {
    {
      static int *srccells[26];
      for (int i = 0; i < 26; ++i) srccells[i] = sendhalos[i].dcellstarts.D;

      CC(cudaMemcpyToSymbol(PackingHalo::srccells, srccells, sizeof(srccells),
                            0, cudaMemcpyHostToDevice));

      static int *dstcells[26];
      for (int i = 0; i < 26; ++i)
        dstcells[i] = sendhalos[i].hcellstarts.devptr;

      CC(cudaMemcpyToSymbol(PackingHalo::dstcells, dstcells, sizeof(dstcells),
                            0, cudaMemcpyHostToDevice));
    }

    {
      static int *srccells[26];
      for (int i = 0; i < 26; ++i)
        srccells[i] = recvhalos[i].hcellstarts.devptr;

      CC(cudaMemcpyToSymbol(PackingHalo::srccells, srccells, sizeof(srccells),
                            sizeof(srccells), cudaMemcpyHostToDevice));

      static int *dstcells[26];
      for (int i = 0; i < 26; ++i) dstcells[i] = recvhalos[i].dcellstarts.D;

      CC(cudaMemcpyToSymbol(PackingHalo::dstcells, dstcells, sizeof(dstcells),
                            sizeof(dstcells), cudaMemcpyHostToDevice));
    }
  }

  if (PackingHalo::ncells)
    PackingHalo::copycells<
      0><<<(PackingHalo::ncells + 127) / 128, 128, 0, stream>>>(
								PackingHalo::ncells);

  _pack_all(p, n, firstpost, stream);
  CC(cudaPeekAtLastError());
}

void SolventExchange::post(Particle *p, int n,
                           cudaStream_t stream, cudaStream_t downloadstream) {
  {
    CC(cudaEventSynchronize(evfillall));

    bool succeeded = true;
    for (int i = 0; i < 26; ++i) {
      int nrequired = required_send_bag_size_host[i];
      bool failed_entry =
	nrequired >
	sendhalos[i]
	.dbuf.capacity; // || nrequired > sendhalos[i].hbuf.capacity;

      if (failed_entry) {
        sendhalos[i].dbuf.resize(nrequired);
        // sendhalos[i].hbuf.resize(nrequired);
        sendhalos[i].scattered_entries.resize(nrequired);
        succeeded = false;
      }
    }

    if (!succeeded) {
      _pack_all(p, n, true, stream);

      CC(cudaEventSynchronize(evfillall));
    }

    for (int i = 0; i < 26; ++i) {
      int nrequired = required_send_bag_size_host[i];

      sendhalos[i].dbuf.S = nrequired;
      sendhalos[i].hbuf.resize(nrequired);
      sendhalos[i].scattered_entries.S = nrequired;
    }
  }

  for (int i = 0; i < 26; ++i)
    if (sendhalos[i].hbuf.size)
      CC(cudaMemcpyAsync(sendhalos[i].hbuf.data, sendhalos[i].dbuf.D,
                         sizeof(Particle) * sendhalos[i].hbuf.size,
                         cudaMemcpyDeviceToHost, downloadstream));

  CC(cudaStreamSynchronize(downloadstream));
  {
    for (int i = 0, c = 0; i < 26; ++i)
      if (sendhalos[i].expected)
        MC(MPI_Isend(sendhalos[i].hcellstarts.data,
		     sendhalos[i].hcellstarts.size, MPI_INTEGER,
		     dstranks[i], basetag + i + 350, cartcomm,
		     sendcellsreq + c++));

    for (int i = 0, c = 0; i < 26; ++i)
      if (sendhalos[i].expected)
        MC(MPI_Isend(&sendhalos[i].hbuf.size, 1, MPI_INTEGER,
		     dstranks[i], basetag + i + 150, cartcomm,
		     sendcountreq + c++));

    nsendreq = 0;

    for (int i = 0; i < 26; ++i) {
      int expected = sendhalos[i].expected;

      if (expected == 0) continue;

      int count = sendhalos[i].hbuf.size;

      MC(MPI_Isend(sendhalos[i].hbuf.data, expected,
		   Particle::datatype(), dstranks[i], basetag + i,
		   cartcomm, sendreq + nsendreq));

      ++nsendreq;

      if (count > expected) {

        int difference = count - expected;

        int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
        printf("extra message from rank %d to rank %d in the direction of %d "
               "%d %d! difference %d, expected is %d\n",
               myrank, dstranks[i], d[0], d[1], d[2], difference, expected);

        MC(MPI_Isend(sendhalos[i].hbuf.data + expected, difference,
		     Particle::datatype(), dstranks[i],
		     basetag + i + 555, cartcomm, sendreq + nsendreq));
        ++nsendreq;
      }
    }
  }
  firstpost = false;
}

void SolventExchange::post_expected_recv() {
  for (int i = 0, c = 0; i < 26; ++i) {
    if (recvhalos[i].expected)
      MC(MPI_Irecv(recvhalos[i].hbuf.data, recvhalos[i].expected,
		   Particle::datatype(), dstranks[i],
		   basetag + recv_tags[i], cartcomm, recvreq + c++));
  }
  for (int i = 0, c = 0; i < 26; ++i)
    if (recvhalos[i].expected)
      MC(MPI_Irecv(recvhalos[i].hcellstarts.data,
		   recvhalos[i].hcellstarts.size, MPI_INTEGER,
		   dstranks[i], basetag + recv_tags[i] + 350, cartcomm,
		   recvcellsreq + c++));

  for (int i = 0, c = 0; i < 26; ++i)
    if (recvhalos[i].expected)
      MC(MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i],
		   basetag + recv_tags[i] + 150, cartcomm,
		   recvcountreq + c++));
    else
      recv_counts[i] = 0;
}

void SolventExchange::recv(cudaStream_t stream, cudaStream_t uploadstream) {
  CC(cudaPeekAtLastError());
  {
    MPI_Status statuses[26];

    MC(MPI_Waitall(nactive, recvreq, statuses));
    MC(MPI_Waitall(nactive, recvcellsreq, statuses));
    MC(MPI_Waitall(nactive, recvcountreq, statuses));
  }

  for (int i = 0; i < 26; ++i) {
    int count = recv_counts[i];
    int expected = recvhalos[i].expected;
    int difference = count - expected;

    if (count <= expected) {
      recvhalos[i].hbuf.resize(count);
      recvhalos[i].dbuf.resize(count);
    } else {
      printf("RANK %d waiting for RECV-extra message: count %d expected %d "
             "(difference %d) from rank %d\n",
             myrank, count, expected, difference, dstranks[i]);
      recvhalos[i].hbuf.preserve_resize(count);
      recvhalos[i].dbuf.resize(count);
      MPI_Status status;
      MPI_Recv(recvhalos[i].hbuf.data + expected, difference,
               Particle::datatype(), dstranks[i], basetag + recv_tags[i] + 555,
               cartcomm, &status);
    }
  }

  for (int i = 0; i < 26; ++i)
    CC(cudaMemcpyAsync(recvhalos[i].dbuf.D, recvhalos[i].hbuf.data,
                       sizeof(Particle) * recvhalos[i].hbuf.size,
                       cudaMemcpyHostToDevice, uploadstream));

  for (int i = 0; i < 26; ++i)
    CC(cudaMemcpyAsync(recvhalos[i].dcellstarts.D,
                       recvhalos[i].hcellstarts.data,
                       sizeof(int) * recvhalos[i].hcellstarts.size,
                       cudaMemcpyHostToDevice, uploadstream));

  CC(cudaPeekAtLastError());
  post_expected_recv();
}

int SolventExchange::nof_sent_particles() {
  int s = 0;
  for (int i = 0; i < 26; ++i) s += sendhalos[i].hbuf.size;
  return s;
}

void SolventExchange::_cancel_recv() {
  if (!firstpost) {
    {
      MPI_Status statuses[26 * 2];
      MC(MPI_Waitall(nactive, sendcellsreq, statuses));
      MC(MPI_Waitall(nsendreq, sendreq, statuses));
      MC(MPI_Waitall(nactive, sendcountreq, statuses));
    }

    for (int i = 0; i < nactive; ++i) MC(MPI_Cancel(recvreq + i));
    for (int i = 0; i < nactive; ++i) MC(MPI_Cancel(recvcellsreq + i));
    for (int i = 0; i < nactive; ++i) MC(MPI_Cancel(recvcountreq + i));
    firstpost = true;
  }
}

void SolventExchange::adjust_message_sizes(ExpectedMessageSizes sizes) {
  _cancel_recv();
  nactive = 0;
  for (int i = 0; i < 26; ++i) {
    int d[3] = {(i + 2) % 3, (i / 3 + 2) % 3, (i / 9 + 2) % 3};
    int entry = d[0] + 3 * (d[1] + 3 * d[2]);
    int estimate = sizes.msgsizes[entry] * safety_factor;
    estimate = 64 * ((estimate + 63) / 64);
    recvhalos[i].adjust(estimate);
    sendhalos[i].adjust(estimate);
    if (estimate == 0) required_send_bag_size_host[i] = 0;
    nactive += (int)(estimate > 0);
  }
}

SolventExchange::~SolventExchange() {
  CC(cudaFreeHost(required_send_bag_size));
  MC(MPI_Comm_free(&cartcomm));
  _cancel_recv();
  CC(cudaEventDestroy(evfillall));
  CC(cudaEventDestroy(evdownloaded));
}
