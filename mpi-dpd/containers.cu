/*
 *  containers.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-05.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <sys/stat.h>

#include <rbc-cuda.h>
#include <vector>
#include <string>

#include <cstdio>
#include <mpi.h>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"
#include "containers.h"
#include "io.h"
#include "last_bit_float.h"
#include "dpd-forces.h"

int (*CollectionRBC::indices)[3] = NULL, CollectionRBC::ntriangles = -1, CollectionRBC::nvertices = -1;

namespace ParticleKernels
{
    __global__ void upd_stg1(bool rbcflag, Particle * p, Acceleration * a, int n, float dt,
			     float _driving_acceleration, float threshold, bool doublePoiseuille) {
	int pid = threadIdx.x + blockDim.x * blockIdx.x;
	if (pid >= n) return;

	int type; float driving_acceleration, mass, vx = p[pid].u[0], y = p[pid].x[1];
	if      ( rbcflag                            ) type = MEMB_TYPE;
	else if (!rbcflag &&  last_bit_float::get(vx)) type =   IN_TYPE;
	else if (!rbcflag && !last_bit_float::get(vx)) type =  OUT_TYPE;
	mass                 = (type == MEMB_TYPE) ? rbc_mass : 1;
	driving_acceleration = (type ==   IN_TYPE) ? 0        : _driving_acceleration;
	if (doublePoiseuille && y <= threshold) driving_acceleration *= -1;

	for(int c = 0; c < 3; ++c) {
	  last_bit_float::Preserver up0(p[pid].u[0]);
	  p[pid].u[c] += (a[pid].a[c]/mass + (c == 0 ? driving_acceleration : 0)) * dt * 0.5;
	}
	for(int c = 0; c < 3; ++c) p[pid].x[c] += p[pid].u[c] * dt;
    }

    __global__ void upd_stg2_and_1(bool rbcflag, float2 * _pdata, float * _adata,
					int nparticles, float dt, float _driving_acceleration, float threshold,
				   bool doublePoiseuille) {
#if !defined(__CUDA_ARCH__)
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif
	int warpid = threadIdx.x >> 5;
	int base = 32 * (warpid + 4 * blockIdx.x);
	int nsrc = min(32, nparticles - base);

	float2 *pdata = _pdata + 3 * base;
	float *adata = _adata + 3 * base;

	int laneid;
	asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

	int nwords = 3 * nsrc;

	float2 s0, s1, s2;
	float ax, ay, az;

	if (laneid < nwords)
	{
	    s0 = _ACCESS(pdata + laneid);
	    ax = _ACCESS(adata + laneid);
	}

	if (laneid + 32 < nwords)
	{
	    s1 = _ACCESS(pdata + laneid + 32);
	    ay = _ACCESS(adata + laneid + 32);
	}

	if (laneid + 64 < nwords)
	{
	    s2 = _ACCESS(pdata + laneid + 64);
	    az = _ACCESS(adata + laneid + 64);
	}

	{
	    int srclane0 = (3 * laneid + 0) & 0x1f;
	    int srclane1 = (srclane0 + 1) & 0x1f;
	    int srclane2 = (srclane0 + 2) & 0x1f;

	    int start = laneid % 3;

	    {
		float t0 = __shfl(start == 0 ? s0.x : start == 1 ? s1.x : s2.x, srclane0);
		float t1 = __shfl(start == 0 ? s2.x : start == 1 ? s0.x : s1.x, srclane1);
		float t2 = __shfl(start == 0 ? s1.x : start == 1 ? s2.x : s0.x, srclane2);

		s0.x = t0;
		s1.x = t1;
		s2.x = t2;
	    }

	    {
		float t0 = __shfl(start == 0 ? s0.y : start == 1 ? s1.y : s2.y, srclane0);
		float t1 = __shfl(start == 0 ? s2.y : start == 1 ? s0.y : s1.y, srclane1);
		float t2 = __shfl(start == 0 ? s1.y : start == 1 ? s2.y : s0.y, srclane2);

		s0.y = t0;
		s1.y = t1;
		s2.y = t2;
	    }

	    {
		float t0 = __shfl(start == 0 ? ax : start == 1 ? ay : az, srclane0);
		float t1 = __shfl(start == 0 ? az : start == 1 ? ax : ay, srclane1);
		float t2 = __shfl(start == 0 ? ay : start == 1 ? az : ax, srclane2);

		ax = t0;
		ay = t1;
		az = t2;
	    }
	}

	int type; float driving_acceleration, mass, vx = s1.y, y = s0.y;
	if      (rbcflag                             ) type = MEMB_TYPE;
	else if (!rbcflag &&  last_bit_float::get(vx)) type =  IN_TYPE;
	else if (!rbcflag && !last_bit_float::get(vx)) type = OUT_TYPE;
	mass                 = (type == MEMB_TYPE) ? rbc_mass : 1;
	driving_acceleration = (type ==   IN_TYPE) ? 0        : _driving_acceleration;
	if (doublePoiseuille && y <= threshold) driving_acceleration *= -1;

	s1.y += (ax/mass + driving_acceleration) * dt;
	s2.x += ay/mass * dt;
	s2.y += az/mass * dt;

	s0.x += s1.y * dt;
	s0.y += s2.x * dt;
	s1.x += s2.y * dt;

	{
	    int srclane0 = (32 * ((laneid) % 3) + laneid) / 3;
	    int srclane1 = (32 * ((laneid + 1) % 3) + laneid) / 3;
	    int srclane2 = (32 * ((laneid + 2) % 3) + laneid) / 3;

	    int start = laneid % 3;

	    {
		float t0 = __shfl(s0.x, srclane0);
		float t1 = __shfl(s2.x, srclane1);
		float t2 = __shfl(s1.x, srclane2);

		s0.x = start == 0 ? t0 : start == 1 ? t2 : t1;
		s1.x = start == 0 ? t1 : start == 1 ? t0 : t2;
		s2.x = start == 0 ? t2 : start == 1 ? t1 : t0;
	    }

	    {
		float t0 = __shfl(s0.y, srclane0);
		float t1 = __shfl(s2.y, srclane1);
		float t2 = __shfl(s1.y, srclane2);

		s0.y = start == 0 ? t0 : start == 1 ? t2 : t1;
		s1.y = start == 0 ? t1 : start == 1 ? t0 : t2;
		s2.y = start == 0 ? t2 : start == 1 ? t1 : t0;
	    }

	    {
		float t0 = __shfl(ax, srclane0);
		float t1 = __shfl(az, srclane1);
		float t2 = __shfl(ay, srclane2);

		ax = start == 0 ? t0 : start == 1 ? t2 : t1;
		ay = start == 0 ? t1 : start == 1 ? t0 : t2;
		az = start == 0 ? t2 : start == 1 ? t1 : t0;
	    }
	}

	if (laneid < nwords) {
	    last_bit_float::Preserver up1(pdata[laneid].y);
	    pdata[laneid] = s0;
	}

	if (laneid + 32 < nwords) {
	    last_bit_float::Preserver up1(pdata[laneid + 32].y);
	    pdata[laneid + 32] = s1;
	}

	if (laneid + 64 < nwords) {
	    last_bit_float::Preserver up1(pdata[laneid + 64].y);
	    pdata[laneid + 64] = s2;
	}
    }

    __global__ void clear_velocity(Particle *p, int n)  {
	int pid = threadIdx.x + blockDim.x * blockIdx.x;
	if (pid >= n) return;
	last_bit_float::Preserver up(p[pid].u[0]);
	for(int c = 0; c < 3; ++c) p[pid].u[c] = 0;
    }
} /* end of ParticleKernels */

void ParticleArray::upd_stg1(bool rbcflag, float driving_acceleration, cudaStream_t stream) {
    if (size)
	ParticleKernels::upd_stg1<<<(xyzuvw.size + 127) / 128, 128, 0, stream>>>
	  (rbcflag, xyzuvw.data, axayaz.data, xyzuvw.size,
	   dt, driving_acceleration, globalextent.y * 0.5 - origin.y, doublepoiseuille);
}

void  ParticleArray::upd_stg2_and_1(bool rbcflag, float driving_acceleration, cudaStream_t stream) {
    if (size)
	ParticleKernels::upd_stg2_and_1<<<(xyzuvw.size + 127) / 128, 128, 0, stream>>>
	  (rbcflag, (float2 *)xyzuvw.data, (float *)axayaz.data, xyzuvw.size,
	   dt, driving_acceleration, globalextent.y * 0.5 - origin.y, doublepoiseuille);
}

void ParticleArray::resize(int n) {
    size = n;

    // YTANG: need the array to be 32-padded for locally transposed storage of acceleration
    if ( n % 32 ) {
	xyzuvw.preserve_resize( n - n % 32 + 32 );
	axayaz.preserve_resize( n - n % 32 + 32 );
    }
    xyzuvw.resize(n);
    axayaz.resize(n);

#ifndef NDEBUG
    CUDA_CHECK(cudaMemset(axayaz.data, 0, sizeof(Acceleration) * size));
#endif
}

void ParticleArray::preserve_resize(int n) {
    int oldsize = size;
    size = n;

    xyzuvw.preserve_resize(n);
    axayaz.preserve_resize(n);

    if (size > oldsize)
	CUDA_CHECK(cudaMemset(axayaz.data + oldsize, 0, sizeof(Acceleration) * (size-oldsize)));
}

void ParticleArray::clear_velocity() {
    if (size)
	ParticleKernels::clear_velocity<<<(xyzuvw.size + 127) / 128, 128 >>>(xyzuvw.data, xyzuvw.size);
}

void CollectionRBC::resize(int count) {
    ncells = count;
    ParticleArray::resize(count * get_nvertices());
}

void CollectionRBC::preserve_resize(int count) {
    ncells = count;
    ParticleArray::preserve_resize(count * get_nvertices());
}

CollectionRBC::CollectionRBC(MPI_Comm cartcomm_) {
  cartcomm = cartcomm_; ncells = 0;
  MPI_CHECK(MPI_Comm_rank(cartcomm, &myrank));
  MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

  CudaRBC::get_triangle_indexing(indices, ntriangles);
  CudaRBC::Extent extent;
  CudaRBC::setup(nvertices, extent);
}

struct TransformedExtent {
    float com[3];
    float transform[4][4];
};
void CollectionRBC::setup(const char *path2ic) {
    vector<TransformedExtent> allrbcs;
    if (myrank == 0) {
	//read transformed extent from file
	FILE *f = fopen(path2ic, "r");
	printf("READING FROM: <%s>\n", path2ic);
	bool isgood = true;
	while(isgood) {
	    float tmp[19];
	    for(int c = 0; c < 19; ++c) {
		int retval = fscanf(f, "%f", tmp + c);
		isgood &= retval == 1;
	    }

	    if (isgood) {
		TransformedExtent t;

		for(int c = 0; c < 3; ++c) t.com[c] = tmp[c];

		int ctr = 3;
		for(int c = 0; c < 16; ++c, ++ctr) t.transform[c / 4][c % 4] = tmp[ctr];
		allrbcs.push_back(t);
	    }
	}
	fclose(f);
	printf("Instantiating %d CELLs from...<%s>\n", (int)allrbcs.size(), path2ic);
    } /* end of myrank == 0 */

    int allrbcs_count = allrbcs.size();
    MPI_CHECK(MPI_Bcast(&allrbcs_count, 1, MPI_INT, 0, cartcomm));

    allrbcs.resize(allrbcs_count);

    int nfloats_per_entry = sizeof(TransformedExtent) / sizeof(float);

    MPI_CHECK(MPI_Bcast(&allrbcs.front(), nfloats_per_entry * allrbcs_count, MPI_FLOAT, 0, cartcomm));

    vector<TransformedExtent> good;

    int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

    for(vector<TransformedExtent>::iterator it = allrbcs.begin(); it != allrbcs.end(); ++it) {
	bool inside = true;
	for(int c = 0; c < 3; ++c)
	    inside &= it->com[c] >= coords[c] * L[c] && it->com[c] < (coords[c] + 1) * L[c];
	if (inside) {
	    for(int c = 0; c < 3; ++c)
		it->transform[c][3] -= (coords[c] + 0.5) * L[c];
	    good.push_back(*it);
	}
    }

    resize(good.size());
    for(int i = 0; i < good.size(); ++i)
	_initialize((float *)(xyzuvw.data + get_nvertices() * i), good[i].transform);
}

void CollectionRBC::_initialize(float *device_xyzuvw, float (*transform)[4]) {
    CudaRBC::initialize(device_xyzuvw, transform);
}

void CollectionRBC::remove(int *entries, int nentries) {
    std::vector<bool > marks(ncells, true);

    for(int i = 0; i < nentries; ++i)
	marks[entries[i]] = false;

    std::vector< int > survivors;
    for(int i = 0; i < ncells; ++i)
	if (marks[i])
	    survivors.push_back(i);

    int nsurvived = survivors.size();

    SimpleDeviceBuffer<Particle> survived(get_nvertices() *nsurvived);

    for(int i = 0; i < nsurvived; ++i)
	CUDA_CHECK(cudaMemcpy(survived.data + get_nvertices() * i, data() + get_nvertices() * survivors[i],
		    sizeof(Particle) * get_nvertices(), cudaMemcpyDeviceToDevice));

    resize(nsurvived);

    CUDA_CHECK(cudaMemcpy(xyzuvw.data, survived.data, sizeof(Particle) * survived.size, cudaMemcpyDeviceToDevice));
}

void CollectionRBC::_dump(const char *format4ply,
			  MPI_Comm comm, MPI_Comm cartcomm, int ntriangles, int ncells, int nvertices, int (* const indices)[3],
			  Particle *p, Acceleration *a, int n, int iddatadump) {
    int ctr = iddatadump;

    //we fused VV stages so we need to recover the state before stage 1
    for(int i = 0; i < n; ++i) {
	last_bit_float::Preserver up(p[i].u[0]);
	for(int c = 0; c < 3; ++c) {
	    p[i].x[c] -= dt * p[i].u[c];
	    p[i].u[c] -= 0.5 * dt * a[i].a[c];
	}
    }
    char buf[200];
    sprintf(buf, format4ply, ctr);

    int rank;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));

    if(rank == 0)
      mkdir("ply", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    ply_dump(comm, cartcomm, buf, indices, ncells, ntriangles, p, nvertices, false);
}
