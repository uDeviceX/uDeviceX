namespace k_rex {
  __constant__ int ccapacities[26], *scattered_indices[26];
  __device__ bool failed;
  __constant__ int coffsets[26];
  __constant__ int ccounts[26], cbases[27], cpaddedstarts[27];
  __constant__ float *recvbags[26];


__global__ void init() { failed = false; }
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

__global__ void pack(float2 *particles, int nparticles,
		     float2 *buffer, int nbuffer,
		     int soluteid) {
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
      int pid = __ldg(scattered_indices[code] + entry);

      int entry2 = 3 * pid;

      s0 = __ldg(particles + entry2);
      s1 = __ldg(particles + entry2 + 1);
      s2 = __ldg(particles + entry2 + 2);

      s0.x -= ((code + 2) % 3 - 1) * XSIZE_SUBDOMAIN;
      s0.y -= ((code / 3 + 2) % 3 - 1) * YSIZE_SUBDOMAIN;
      s1.x -= ((code / 9 + 2) % 3 - 1) * ZSIZE_SUBDOMAIN;
    }
    write_AOS6f(buffer + 3 * (cbases[code] + coffsets[code] + packbase), npack,
		s0, s1, s2);
  }
}

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
    float myval = __ldg(recvbags[code] + component + 3 * entry);
    int dpid = __ldg(scattered_indices[code] + entry);

    atomicAdd(accelerations + 3 * dpid + component, myval);
  }
}
}
