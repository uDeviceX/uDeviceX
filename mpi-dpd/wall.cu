/*
 *  wall.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-19.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <sys/stat.h>
#include <sys/types.h>

#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <mpi.h>
#include "common.h"
#include "io.h"
#include <dpd-rng.h>
#include "wall.h"
#include "redistancing.h"
#include "dpd-forces.h"
#include "last_bit_float.h"

enum {
  XSIZE_WALLCELLS = 2 * XMARGIN_WALL + XSIZE_SUBDOMAIN,
  YSIZE_WALLCELLS = 2 * YMARGIN_WALL + YSIZE_SUBDOMAIN,
  ZSIZE_WALLCELLS = 2 * ZMARGIN_WALL + ZSIZE_SUBDOMAIN,

  XTEXTURESIZE = 256,

  _YTEXTURESIZE = ((YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL) * XTEXTURESIZE +
		   XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL - 1) /
  (XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL),

  YTEXTURESIZE = 16 * ((_YTEXTURESIZE + 15) / 16),

  _ZTEXTURESIZE = ((ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL) * XTEXTURESIZE +
		   XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL - 1) /
  (XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL),

  ZTEXTURESIZE = 16 * ((_ZTEXTURESIZE + 15) / 16),

};

namespace SolidWallsKernel {
  texture<float, 3, cudaReadModeElementType> texSDF;

  texture<float4, 1, cudaReadModeElementType> texWallParticles;
  texture<int, 1, cudaReadModeElementType> texWallCellStart, texWallCellCount;

  __global__ void interactions_3tpp(const float2 *const pp, const int np,
				    const int nsolid, float *const acc,
				    const float seed);
  void setup() {
    texSDF.normalized = 0;
    texSDF.filterMode = cudaFilterModePoint;
    texSDF.mipmapFilterMode = cudaFilterModePoint;
    texSDF.addressMode[0] = cudaAddressModeWrap;
    texSDF.addressMode[1] = cudaAddressModeWrap;
    texSDF.addressMode[2] = cudaAddressModeWrap;

    texWallParticles.channelDesc = cudaCreateChannelDesc<float4>();
    texWallParticles.filterMode = cudaFilterModePoint;
    texWallParticles.mipmapFilterMode = cudaFilterModePoint;
    texWallParticles.normalized = 0;

    texWallCellStart.channelDesc = cudaCreateChannelDesc<int>();
    texWallCellStart.filterMode = cudaFilterModePoint;
    texWallCellStart.mipmapFilterMode = cudaFilterModePoint;
    texWallCellStart.normalized = 0;

    texWallCellCount.channelDesc = cudaCreateChannelDesc<int>();
    texWallCellCount.filterMode = cudaFilterModePoint;
    texWallCellCount.mipmapFilterMode = cudaFilterModePoint;
    texWallCellCount.normalized = 0;

    CUDA_CHECK(cudaFuncSetCacheConfig(interactions_3tpp, cudaFuncCachePreferL1));
  }

  __device__ float sdf(float x, float y, float z) {
    int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
    int MARGIN[3] = {XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL};
    int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};

    float tc[3], lmbd[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c) {
      float t =
	TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]);

      lmbd[c] = t - (int)t;
      tc[c] = (int)t + 0.5;
    }
#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    float s000 = tex0(0, 0, 0), s001 = tex0(1, 0, 0), s010 = tex0(0, 1, 0);
    float s011 = tex0(1, 1, 0), s100 = tex0(0, 0, 1), s101 = tex0(1, 0, 1);
    float s110 = tex0(0, 1, 1), s111 = tex0(1, 1, 1);
#undef tex0

#define wavrg(A, B, p) A*(1-p) + p*B /* weighted average */
    float s00x = wavrg(s000, s001, lmbd[0]);
    float s01x = wavrg(s010, s011, lmbd[0]);
    float s10x = wavrg(s100, s101, lmbd[0]);
    float s11x = wavrg(s110, s111, lmbd[0]);

    float s0yx = wavrg(s00x, s01x, lmbd[1]);

    float s1yx = wavrg(s10x, s11x, lmbd[1]);
    float szyx = wavrg(s0yx, s1yx, lmbd[2]);
#undef wavrg
    return szyx;
  }

  __device__ float cheap_sdf(float x, float y, float z) // within the
							// rescaled
							// texel width
							// error
  {
    int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
    int MARGIN[3] = {XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL};
    int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};

    float tc[3], r[3] = {x, y, z};;
    for (int c = 0; c < 3; ++c)
      tc[c] = 0.5001f + (int)(TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) /
			      (L[c] + 2 * MARGIN[c]));
#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    return tex0(0, 0, 0);
#undef  tex0
  }

  __device__ float3 ugrad_sdf(float x, float y, float z) {
    int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
    int MARGIN[3] = {XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL};
    int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};

    float tc[3], fcts[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
      tc[c] = 0.5001f + (int)(TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) /
			      (L[c] + 2 * MARGIN[c]));
    for (int c = 0; c < 3; ++c) fcts[c] = TEXSIZES[c] / (2 * MARGIN[c] + L[c]);

#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    float myval = tex0(0, 0, 0);
    float gx = fcts[0] * (tex0(1, 0, 0) - myval);
    float gy = fcts[1] * (tex0(0, 1, 0) - myval);
    float gz = fcts[2] * (tex0(0, 0, 1) - myval);
#undef tex0

    return make_float3(gx, gy, gz);
  }

  __device__ float3 grad_sdf(float x, float y, float z) {
    int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
    int MARGIN[3] = {XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL};
    int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};

    float tc[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
      tc[c] =
	TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]);

    float gx, gy, gz;
#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    gx = tex0(1, 0, 0) - tex0(-1,  0,  0);
    gy = tex0(0, 1, 0) - tex0( 0, -1,  0);
    gz = tex0(0, 0, 1) - tex0( 0,  0, -1);
#undef tex0

    float ggmag = sqrt(gx*gx + gy*gy + gz*gz);

    if (ggmag > 1e-6) {
      gx /= ggmag; gy /= ggmag; gz /= ggmag;
    }
    return make_float3(gx, gy, gz);
  }

  __global__ void fill_keys(const Particle *const pp, const int n,
			    int *const key) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= n) return;

    Particle p = pp[pid];

    float sdf0 = sdf(p.x[0], p.x[1], p.x[2]);
    key[pid] = (int)(sdf0 >= 0) + (int)(sdf0 > 2);
  }

  __global__ void strip_solid4(Particle *const src, const int n, float4 *dst) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = src[pid];
    dst[pid] = make_float4(p.x[0], p.x[1], p.x[2], 0);
  }

  __device__ void wall_vell(float x, float y, float z,
			    float *vxw, float *vyw, float *vzw) {
    *vxw = gamma_dot * z; *vyw = 0; *vzw = 0; /* velocity of the wall;
						 TODO: works only for
						 one processor */
  }

  __device__ void bounce_vel(float    x, float    y, float    z,
			     float* vxp, float* vyp, float* vzp) {
    float vx = *vxp,  vy = *vyp, vz = *vzp;

    float vxw, vyw, vzw; wall_vell(x, y, z, &vxw, &vyw, &vzw);

    vx -= vxw; vx = -vx; vx += vxw;
    vy -= vyw; vy = -vy; vy += vyw;
    vz -= vzw; vz = -vz; vz += vzw;

    last_bit_float::Preserver up1(*vxp);
    *vxp = vx; *vyp = vy; *vzp = vz;
  }

  __device__ void handle_collision(float currsdf,
				   float &x, float &y, float &z,
				   float &vx, float &vy, float &vz,
				   float dt) {
    float x0 = x - vx*dt, y0 = y - vy*dt, z0 = z - vz*dt;
    if (sdf(x0, y0, z0) >= 0) {
      // this is the worst case - it means that 0 position was bad already
      // we need to search and rescue the particle
      float3 gg = grad_sdf(x, y, z);
      float sdf0 = currsdf;
      x -= sdf0 * gg.x; y -= sdf0 * gg.y; z -= sdf0 * gg.z;
      for (int l = 8; l >= 1; --l) {
	if (sdf(x, y, z) < 0) {
	  bounce_vel(x, y, z, &vx, &vy, &vz); return;
	}
	float jump = 1.1f * sdf0 / (1 << l);
	x -= jump * gg.x; y -= jump * gg.y; z -= jump * gg.z;
      }
    }

    // newton raphson steps
    float subdt = dt;
    {
      float3 gg = ugrad_sdf(x, y, z);
      float DphiDt = max(1e-4f, gg.x * vx + gg.y * vy + gg.z * vz);
      subdt = min(dt, max(0.f, subdt - currsdf / DphiDt * 1.02f));
    }

    {
      float3 xstar = make_float3(x + subdt * vx, y + subdt * vy, z + subdt * vz);
      float3 gg = ugrad_sdf(xstar.x, xstar.y, xstar.z);
      float DphiDt = max(1e-4f, gg.x * vx + gg.y * vy + gg.z * vz);
      subdt = min(
		  dt, max(0.f, subdt - sdf(xstar.x, xstar.y, xstar.z) / DphiDt * 1.02f));
    }

    float lmbd = 2 * subdt - dt;
    x = x0 + lmbd * vx; y = y0 + lmbd * vy; z = z0 + lmbd * vz;
    bounce_vel(x, y, z, &vx, &vy, &vz);
    if (sdf(x, y, z) >= 0) {
      x = x0; y = y0; z = z0;
    }
  }

  __global__ void bounce(float2 *const pp, int nparticles, float dt) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= nparticles) return;

    float2 data0 = pp[pid * 3];
    float2 data1 = pp[pid * 3 + 1];
    if (pid < nparticles) {
      float mycheapsdf = cheap_sdf(data0.x, data0.y, data1.x);

      if (mycheapsdf >=
	  -1.7320f * ((float)XSIZE_WALLCELLS / (float)XTEXTURESIZE)) {
	float currsdf = sdf(data0.x, data0.y, data1.x);

	float2 data2 = pp[pid * 3 + 2];

	float3 v0 = make_float3(data1.y, data2.x, data2.y);

	if (currsdf >= 0) {
	  handle_collision(currsdf, data0.x, data0.y, data1.x, data1.y, data2.x,
			   data2.y, dt);

	  pp[3 * pid] = data0;
	  pp[3 * pid + 1] = data1;
	  pp[3 * pid + 2] = data2;
	}
      }
    }
  }

  __global__ void interactions_3tpp(const float2 *const pp, const int np,
				    const int nsolid, float *const acc,
				    const float seed) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int pid = gid / 3;
    int zplane = gid % 3;

    if (pid >= np) return;

    float2 dst0 = pp[3 * pid + 0];
    float2 dst1 = pp[3 * pid + 1];

    float interacting_threshold =
      -1 - 1.7320f * ((float)XSIZE_WALLCELLS / (float)XTEXTURESIZE);

    if (cheap_sdf(dst0.x, dst0.y, dst1.x) <= interacting_threshold) return;

    float2 dst2 = pp[3 * pid + 2];

    uint scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;

    {
      int xbase = (int)(dst0.x - (-XSIZE_SUBDOMAIN / 2 - XMARGIN_WALL));
      int ybase = (int)(dst0.y - (-YSIZE_SUBDOMAIN / 2 - YMARGIN_WALL));
      int zbase = (int)(dst1.x - (-ZSIZE_SUBDOMAIN / 2 - ZMARGIN_WALL));

      enum {
	XCELLS = XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL,
	YCELLS = YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL,
	ZCELLS = ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL,
	NCELLS = XCELLS * YCELLS * ZCELLS
      };

      int cid0 = xbase - 1 + XCELLS * (ybase - 1 + YCELLS * (zbase - 1 + zplane));

      spidbase = tex1Dfetch(texWallCellStart, cid0);
      int count0 = tex1Dfetch(texWallCellStart, cid0 + 3) - spidbase;

      int cid1 = cid0 + XCELLS;
      deltaspid1 = tex1Dfetch(texWallCellStart, cid1);
      int count1 = tex1Dfetch(texWallCellStart, cid1 + 3) - deltaspid1;

      int cid2 = cid0 + XCELLS * 2;
      deltaspid2 = tex1Dfetch(texWallCellStart, cid2);
      int count2 = cid2 + 3 == NCELLS
	? nsolid
	: tex1Dfetch(texWallCellStart, cid2 + 3) - deltaspid2;

      scan1 = count0;
      scan2 = count0 + count1;
      ncandidates = scan2 + count2;

      deltaspid1 -= scan1;
      deltaspid2 -= scan2;
    }

    float xforce = 0, yforce = 0, zforce = 0;

#define zig x
#define zag y

#define uno x
#define due y
#define tre z

#define mf3 make_float3
#define TYPE_WALL 3
    float  x = dst0.zig,  y = dst0.zag,  z = dst1.zig; /* bulk particle  */
    float vx = dst1.zag, vy = dst2.zig, vz = dst2.zag;

    for (int i = 0; i < ncandidates; ++i) {
      int m1 = (int)(i >= scan1);
      int m2 = (int)(i >= scan2);
      int spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);
      float4 stmp0 = tex1Dfetch(texWallParticles, spid);

      float  xw = stmp0.uno,  yw = stmp0.due,  zw = stmp0.tre; /* wall particle */
      float vxw, vyw, vzw; wall_vell(xw, yw, zw, &vxw, &vyw, &vzw);
      float rnd = Logistic::mean0var1(seed, pid, spid);

      // check for particle types and compute the DPD force
      int type_bulk = last_bit_float::get(vx);
      float3 strength = compute_dpd_force_traced(type_bulk      , TYPE_WALL,
						 mf3(x ,  y,  z), mf3( xw,  yw,  zw),
						 mf3(vx, vy, vz), mf3(vxw, vyw, vzw), rnd);
      xforce += strength.x; yforce += strength.y; zforce += strength.z;
    }
#undef zig
#undef zag

#undef uno
#undef due

#undef tre
#undef mf3
#undef TYPE_WALL

    atomicAdd(acc + 3 * pid + 0, xforce);
    atomicAdd(acc + 3 * pid + 1, yforce);
    atomicAdd(acc + 3 * pid + 2, zforce);
  }
}

template <int k> struct Bspline {
  template <int i> static float eval(float x) {
    return (x - i) / (k - 1) * Bspline<k - 1>::template eval<i>(x) +
      (i + k - x) / (k - 1) * Bspline<k - 1>::template eval<i + 1>(x);
  }
};

template <> struct Bspline<1> {
  template <int i> static float eval(float x) {
    return (float)(i) <= x && x < (float)(i + 1);
  }
};

struct FieldSampler {
  float *data, extent[3];
  int N[3];

  FieldSampler(const char *path, MPI_Comm comm) {
    static size_t CHUNKSIZE = 1 << 25;

    int rank;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));

    if (rank == 0) {
      char header[2048];

      FILE *fh = fopen(path, "rb");

      fread(header, 1, sizeof(header), fh);

      printf("root parsing header\n");
      int retval = sscanf(header, "%f %f %f\n%d %d %d\n", extent + 0,
			  extent + 1, extent + 2, N + 0, N + 1, N + 2);

      if (retval != 6) {
	printf("ooops something went wrong in reading %s.\n", path);
	exit(EXIT_FAILURE);
      }

      printf("broadcasting N\n");
      MPI_CHECK(MPI_Bcast(N, 3, MPI_INT, 0, comm));
      MPI_CHECK(MPI_Bcast(extent, 3, MPI_FLOAT, 0, comm));

      int nvoxels = N[0] * N[1] * N[2];

      data = new float[nvoxels];

      if (data == NULL) {
	printf("ooops bad allocation %s.\n", path);
	exit(EXIT_FAILURE);
      }

      int header_size = 0;

      for (int i = 0; i < sizeof(header); ++i)
	if (header[i] == '\n') {
	  if (header_size > 0) {
	    header_size = i + 1;
	    break;
	  }

	  header_size = i + 1;
	}

      fseek(fh, header_size, SEEK_SET);
      fread(data, sizeof(float), nvoxels, fh);

      fclose(fh);
      for (size_t i = 0; i < nvoxels; i += CHUNKSIZE) {
	size_t s = (i + CHUNKSIZE <= nvoxels) ? CHUNKSIZE : (nvoxels - i);
	MPI_CHECK(MPI_Bcast(data + i, s, MPI_FLOAT, 0, comm));
      }

    } else {
      MPI_CHECK(MPI_Bcast(N, 3, MPI_INT, 0, comm));
      MPI_CHECK(MPI_Bcast(extent, 3, MPI_FLOAT, 0, comm));
      int nvoxels = N[0] * N[1] * N[2];

      data = new float[nvoxels];

      for (size_t i = 0; i < nvoxels; i += CHUNKSIZE) {
	size_t s = (i + CHUNKSIZE <= nvoxels) ? CHUNKSIZE : (nvoxels - i);
	MPI_CHECK(MPI_Bcast(data + i, s, MPI_FLOAT, 0, comm));
      }
    }
  }

  void sample(const float start[3], const float spacing[3], const int nsize[3],
	      float *const output, const float amplitude_rescaling) {
    Bspline<4> bsp;

    for (int iz = 0; iz < nsize[2]; ++iz)
      for (int iy = 0; iy < nsize[1]; ++iy)
	for (int ix = 0; ix < nsize[0]; ++ix) {
	  float x[3] = {start[0] + (ix + 0.5f) * spacing[0] - 0.5f,
			start[1] + (iy + 0.5f) * spacing[1] - 0.5f,
			start[2] + (iz + 0.5f) * spacing[2] - 0.5f};

	  int anchor[3];
	  for (int c = 0; c < 3; ++c) anchor[c] = (int)floor(x[c]);

	  float w[3][4];
	  for (int c = 0; c < 3; ++c)
	    for (int i = 0; i < 4; ++i)
	      w[c][i] = bsp.eval<0>(x[c] - (anchor[c] - 1 + i) + 2);

	  float tmp[4][4];
	  for (int sz = 0; sz < 4; ++sz)
	    for (int sy = 0; sy < 4; ++sy) {
	      float s = 0;

	      for (int sx = 0; sx < 4; ++sx) {
		int l[3] = {sx, sy, sz};

		int g[3];
		for (int c = 0; c < 3; ++c)
		  g[c] = (l[c] - 1 + anchor[c] + N[c]) % N[c];

		s += w[0][sx] * data[g[0] + N[0] * (g[1] + N[1] * g[2])];
	      }

	      tmp[sz][sy] = s;
	    }

	  float partial[4];
	  for (int sz = 0; sz < 4; ++sz) {
	    float s = 0;

	    for (int sy = 0; sy < 4; ++sy) s += w[1][sy] * tmp[sz][sy];

	    partial[sz] = s;
	  }

	  float val = 0;
	  for (int sz = 0; sz < 4; ++sz) val += w[2][sz] * partial[sz];

	  output[ix + nsize[0] * (iy + nsize[1] * iz)] =
	    val * amplitude_rescaling;
	}
  }

  ~FieldSampler() { delete[] data; }
};

ComputeWall::ComputeWall(MPI_Comm cartcomm, Particle *const p, const int n,
			 int &nsurvived, ExpectedMessageSizes &new_sizes)
  : cartcomm(cartcomm), arrSDF(NULL), solid4(NULL), solid_size(0),
    cells(XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL,
	  YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL,
	  ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL) {
  MPI_CHECK(MPI_Comm_rank(cartcomm, &myrank));

  MPI_CHECK(MPI_Cart_get(cartcomm, 3, dims, periods, coords));

  float *field = new float[XTEXTURESIZE * YTEXTURESIZE * ZTEXTURESIZE];

  FieldSampler sampler("sdf.dat", cartcomm);

  int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
  int MARGIN[3] = {XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL};
  int TEXTURESIZE[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};

  if (myrank == 0) printf("sampling the geometry file...\n");

  {
    float start[3], spacing[3];
    for (int c = 0; c < 3; ++c) {
      start[c] = sampler.N[c] * (coords[c] * L[c] - MARGIN[c]) /
	(float)(dims[c] * L[c]);
      spacing[c] = sampler.N[c] * (L[c] + 2 * MARGIN[c]) /
	(float)(dims[c] * L[c]) / (float)TEXTURESIZE[c];
    }

    float amplitude_rescaling = (XSIZE_SUBDOMAIN /*+ 2 * XMARGIN_WALL*/) /
      (sampler.extent[0] / dims[0]);

    sampler.sample(start, spacing, TEXTURESIZE, field, amplitude_rescaling);
  }

  if (myrank == 0) printf("redistancing the geometry field...\n");

  // extra redistancing because margin might exceed the domain
  {
    double dx = (XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL) / (double)XTEXTURESIZE;
    double dy = (YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL) / (double)YTEXTURESIZE;
    double dz = (ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL) / (double)ZTEXTURESIZE;

    redistancing(field, XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE, dx, dy, dz,
		 XTEXTURESIZE * 2);
  }

  if (myrank == 0) printf("estimating geometry-based message sizes...\n");

  {
    for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
	for (int dx = -1; dx <= 1; ++dx) {
	  int d[3] = {dx, dy, dz};
	  int entry = (dx + 1) + 3 * ((dy + 1) + 3 * (dz + 1));

	  int local_start[3] = {d[0] + (d[0] == 1) * (XSIZE_SUBDOMAIN - 2),
				d[1] + (d[1] == 1) * (YSIZE_SUBDOMAIN - 2),
				d[2] + (d[2] == 1) * (ZSIZE_SUBDOMAIN - 2)};

	  int local_extent[3] = {1 * (d[0] != 0 ? 2 : XSIZE_SUBDOMAIN),
				 1 * (d[1] != 0 ? 2 : YSIZE_SUBDOMAIN),
				 1 * (d[2] != 0 ? 2 : ZSIZE_SUBDOMAIN)};

	  float start[3], spacing[3];
	  for (int c = 0; c < 3; ++c) {
	    start[c] = (coords[c] * L[c] + local_start[c]) /
	      (float)(dims[c] * L[c]) * sampler.N[c];
	    spacing[c] = sampler.N[c] / (float)(dims[c] * L[c]);
	  }

	  int nextent = local_extent[0] * local_extent[1] * local_extent[2];
	  float *data = new float[nextent];

	  sampler.sample(start, spacing, local_extent, data, 1);

	  int s = 0;
	  for (int i = 0; i < nextent; ++i) s += (data[i] < 0);

	  delete[] data;
	  double avgsize =
	    ceil(s * numberdensity /
		 (double)pow(2, abs(d[0]) + abs(d[1]) + abs(d[2])));

	  new_sizes.msgsizes[entry] = (int)avgsize;
	}
  }

  if (hdf5field_dumps) {
    if (myrank == 0) printf("H5 data dump of the geometry...\n");

    float *walldata =
      new float[XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN];

    float start[3], spacing[3];
    for (int c = 0; c < 3; ++c) {
      start[c] = coords[c] * L[c] / (float)(dims[c] * L[c]) * sampler.N[c];
      spacing[c] = sampler.N[c] / (float)(dims[c] * L[c]);
    }

    int size[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};

    float amplitude_rescaling = L[0] / (sampler.extent[0] / dims[0]);
    sampler.sample(start, spacing, size, walldata, amplitude_rescaling);

    H5FieldDump dump(cartcomm);
    dump.dump_scalarfield(cartcomm, walldata, "wall");

    delete[] walldata;
  }

  CUDA_CHECK(cudaPeekAtLastError());

  cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaMalloc3DArray(
			       &arrSDF, &fmt,
			       make_cudaExtent(XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE)));

  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr = make_cudaPitchedPtr(
					  (void *)field, XTEXTURESIZE * sizeof(float), XTEXTURESIZE, YTEXTURESIZE);
  copyParams.dstArray = arrSDF;
  copyParams.extent = make_cudaExtent(XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE);
  copyParams.kind = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpy3D(&copyParams));
  delete[] field;

  SolidWallsKernel::setup();

  CUDA_CHECK(cudaBindTextureToArray(SolidWallsKernel::texSDF, arrSDF, fmt));

  if (myrank == 0) printf("carving out wall particles...\n");

  thrust::device_vector<int> keys(n);

  SolidWallsKernel::fill_keys<<<(n + 127) / 128, 128>>>(
							p, n, thrust::raw_pointer_cast(&keys[0]));
  CUDA_CHECK(cudaPeekAtLastError());

  thrust::sort_by_key(keys.begin(), keys.end(),
		      thrust::device_ptr<Particle>(p));

  nsurvived = thrust::count(keys.begin(), keys.end(), 0);

  int nbelt = thrust::count(keys.begin() + nsurvived, keys.end(), 1);

  thrust::device_vector<Particle> solid_local(
					      thrust::device_ptr<Particle>(p + nsurvived),
					      thrust::device_ptr<Particle>(p + nsurvived + nbelt));

  if (hdf5part_dumps) {
    int n = solid_local.size();

    Particle *phost = new Particle[n];

    CUDA_CHECK(cudaMemcpy(phost, thrust::raw_pointer_cast(&solid_local[0]),
			  sizeof(Particle) * n, cudaMemcpyDeviceToHost));

    H5PartDump solid_dump("solid-walls.h5part", cartcomm, cartcomm);
    solid_dump.dump(phost, n);

    delete[] phost;
  }

  // can't use halo-exchanger class because of MARGIN
  // HaloExchanger halo(cartcomm, L, 666);
  // SimpleDeviceBuffer<Particle> solid_remote;
  // halo.exchange(thrust::raw_pointer_cast(&solid_local[0]),
  // solid_local.size(), solid_remote);

  if (myrank == 0)
    printf("fetching remote wall particles in my proximity...\n");

  SimpleDeviceBuffer<Particle> solid_remote;

  {
    thrust::host_vector<Particle> local = solid_local;

    int dstranks[26], remsizes[26], recv_tags[26];
    for (int i = 0; i < 26; ++i) {
      int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

      recv_tags[i] =
	(2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

      int coordsneighbor[3];
      for (int c = 0; c < 3; ++c) coordsneighbor[c] = coords[c] + d[c];

      MPI_CHECK(MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i));
    }

    // send local counts - receive remote counts
    {
      for (int i = 0; i < 26; ++i) remsizes[i] = -1;

      MPI_Request reqrecv[26];
      for (int i = 0; i < 26; ++i)
	MPI_CHECK(MPI_Irecv(remsizes + i, 1, MPI_INTEGER, dstranks[i],
			    123 + recv_tags[i], cartcomm, reqrecv + i));

      int localsize = local.size();

      MPI_Request reqsend[26];
      for (int i = 0; i < 26; ++i)
	MPI_CHECK(MPI_Isend(&localsize, 1, MPI_INTEGER, dstranks[i], 123 + i,
			    cartcomm, reqsend + i));

      MPI_Status statuses[26];
      MPI_CHECK(MPI_Waitall(26, reqrecv, statuses));
      MPI_CHECK(MPI_Waitall(26, reqsend, statuses));
    }

    std::vector<Particle> remote[26];

    // send local data - receive remote data
    {
      for (int i = 0; i < 26; ++i) remote[i].resize(remsizes[i]);

      MPI_Request reqrecv[26];
      for (int i = 0; i < 26; ++i)
	MPI_CHECK(MPI_Irecv(remote[i].data(), remote[i].size() * 6, MPI_FLOAT,
			    dstranks[i], 321 + recv_tags[i], cartcomm,
			    reqrecv + i));

      MPI_Request reqsend[26];
      for (int i = 0; i < 26; ++i)
	MPI_CHECK(MPI_Isend(local.data(), local.size() * 6, MPI_FLOAT,
			    dstranks[i], 321 + i, cartcomm, reqsend + i));

      MPI_Status statuses[26];
      MPI_CHECK(MPI_Waitall(26, reqrecv, statuses));
      MPI_CHECK(MPI_Waitall(26, reqsend, statuses));
    }

    // select particles within my region [-L / 2 - MARGIN, +L / 2 + MARGIN]
    std::vector<Particle> selected;
    for (int i = 0; i < 26; ++i) {
      int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

      for (int j = 0; j < remote[i].size(); ++j) {
	Particle p = remote[i][j];

	for (int c = 0; c < 3; ++c) p.x[c] += d[c] * L[c];

	bool inside = true;

	for (int c = 0; c < 3; ++c)
	  inside &=
	    p.x[c] >= -L[c] / 2 - MARGIN[c] && p.x[c] < L[c] / 2 + MARGIN[c];

	if (inside) selected.push_back(p);
      }
    }

    solid_remote.resize(selected.size());
    CUDA_CHECK(cudaMemcpy(solid_remote.data, selected.data(),
			  sizeof(Particle) * solid_remote.size,
			  cudaMemcpyHostToDevice));
  }

  solid_size = solid_local.size() + solid_remote.size;

  Particle *solid;
  CUDA_CHECK(cudaMalloc(&solid, sizeof(Particle) * solid_size));
  CUDA_CHECK(cudaMemcpy(solid, thrust::raw_pointer_cast(&solid_local[0]),
			sizeof(Particle) * solid_local.size(),
			cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(solid + solid_local.size(), solid_remote.data,
			sizeof(Particle) * solid_remote.size,
			cudaMemcpyDeviceToDevice));

  if (solid_size > 0) cells.build(solid, solid_size, 0);

  CUDA_CHECK(cudaMalloc(&solid4, sizeof(float4) * solid_size));

  if (myrank == 0) printf("consolidating wall particles...\n");

  if (solid_size > 0)
    SolidWallsKernel::strip_solid4<<<(solid_size + 127) / 128, 128>>>(
								      solid, solid_size, solid4);

  CUDA_CHECK(cudaFree(solid));

  CUDA_CHECK(cudaPeekAtLastError());

  frcs.resize(round(1.2 * n / 32.0));
  CUDA_CHECK(cudaMemset(frcs.data, 0, frcs.size * sizeof(float3)));
  samples = 0;
}

void ComputeWall::bounce(Particle *const p, const int n, cudaStream_t stream) {
  if (n > 0)
    SolidWallsKernel::bounce<<<(n + 127) / 128, 128, 0, stream>>>(
								  (float2 *)p, n, dt);

  samples++;
  CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeWall::interactions(const Particle *const p, const int n,
			       Acceleration *const acc,
			       cudaStream_t stream) {
  // cellsstart and cellscount IGNORED for now

  if (n > 0 && solid_size > 0) {
    size_t textureoffset;
    CUDA_CHECK(cudaBindTexture(&textureoffset,
			       &SolidWallsKernel::texWallParticles, solid4,
			       &SolidWallsKernel::texWallParticles.channelDesc,
			       sizeof(float4) * solid_size));

    CUDA_CHECK(cudaBindTexture(&textureoffset,
			       &SolidWallsKernel::texWallCellStart, cells.start,
			       &SolidWallsKernel::texWallCellStart.channelDesc,
			       sizeof(int) * cells.ncells));

    CUDA_CHECK(cudaBindTexture(&textureoffset,
			       &SolidWallsKernel::texWallCellCount, cells.count,
			       &SolidWallsKernel::texWallCellCount.channelDesc,
			       sizeof(int) * cells.ncells));

    SolidWallsKernel::
      interactions_3tpp<<<(3 * n + 127) / 128, 128, 0, stream>>>(
								 (float2 *)p, n, solid_size, (float *)acc, trunk.get_float());

    CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texWallParticles));
    CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texWallCellStart));
    CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texWallCellCount));
  }

  CUDA_CHECK(cudaPeekAtLastError());
}

ComputeWall::~ComputeWall() {
  CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texSDF));
  CUDA_CHECK(cudaFreeArray(arrSDF));
}
