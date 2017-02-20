/*
 *  simulation.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-24.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <sys/stat.h>
#include <map>
#include <string>
#include <vector>
#include <dpd-rng.h>
#include <rbc-cuda.h>
#include <cstdio>
#include <mpi.h>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"
#include "solvent-exchange.h"
#include "dpd.h"
#include "solute-exchange.h"
#include "fsi.h"
#include "contact.h"
#include "redistribute-rbcs.h"
#include "io.h"
#include "simulation.h"
#include "dpd-forces.h"
#include "last_bit_float.h"
#include "geom-wrapper.h"
#include "common-kernels.h"
#include "scan.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

/*** from redistribute-particles.h ***/
int basetag = 950;
MPI_Comm cartcomm_rdst;
float safety_factor = 1.2;
int neighbor_ranks[27], recv_tags[27],
  default_message_sizes[27], send_sizes[27], recv_sizes[27],
  nsendmsgreq, nexpected, nbulk, nhalo, nhalo_padded, myrank;
bool firstcall;
int nactiveneighbors;
MPI_Request sendcountreq[27], recvcountreq[27],
  sendmsgreq[27 * 2], recvmsgreq[27 * 2];
cudaEvent_t evpacking, evsizes;

float * pinnedhost_sendbufs[27], * pinnedhost_recvbufs[27];
struct UnpackBuffer {
  float2 * buffer;
  int capacity;
};

struct PackBuffer {
  float2 * buffer;
  int capacity;
  int * scattered_indices;
};

PackBuffer packbuffers[27];
UnpackBuffer unpackbuffers[27];

PinnedHostBuffer<bool> *failure;
PinnedHostBuffer<int> *packsizes;
SimpleDeviceBuffer<unsigned char> *compressed_cellcounts;
SimpleDeviceBuffer<Particle> *remote_particles;
SimpleDeviceBuffer<uint> *scattered_indices;
SimpleDeviceBuffer<uchar4> *subindices, *subindices_remote;
#include "redistribute-particles.impl.h"

/*** from containters.h ****/
float3 origin, globalextent;
int    coords[3];
MPI_Comm cartcomm;
int nranks, rank;
int ncells = 0;
struct ParticleArray {
  int S; /* size */
  SimpleDeviceBuffer<Particle>     pp; /* xyzuvw */
  SimpleDeviceBuffer<Acceleration> aa; /* axayaz */
};

#include "containers.impl.h"

/**** from wall.h ****/
Logistic::KISS* trunk;

int solid_size;
float4 *solid4;
cudaArray *arrSDF;
CellLists *wall_cells;
#include "wall.impl.h"

/**** from simulation.h ****/
ParticleArray *particles_pingpong[2];
ParticleArray *particles, *newparticles;
ParticleArray *rbcscoll;

SimpleDeviceBuffer<float4 > *xyzouvwo;
SimpleDeviceBuffer<ushort4> *xyzo_half;

CellLists* cells;
RedistributeRBCs* redistribute_rbcs;

ComputeDPD* dpd;
SoluteExchange* solutex;
ComputeFSI* fsi;
ComputeContact* contact;

bool wall_created = false;
bool sim_is_done = false;

MPI_Comm activecomm;
cudaStream_t mainstream, uploadstream, downloadstream;

size_t nsteps;
float driving_acceleration = 0;

pthread_t thread_datadump;
pthread_mutex_t mutex_datadump;
pthread_cond_t request_datadump, done_datadump;
bool datadump_pending = false;
int datadump_idtimestep, datadump_nsolvent, datadump_nrbcs;
bool async_thread_initialized;

PinnedHostBuffer<Particle>      *particles_datadump;
PinnedHostBuffer<Acceleration>  *accelerations_datadump;

cudaEvent_t evdownloaded;

#define NPMAX 5000000 /* TODO: */
float rbc_xx[NPMAX], rbc_yy[NPMAX], rbc_zz[NPMAX];
float sol_xx[NPMAX], sol_yy[NPMAX], sol_zz[NPMAX];
int iotags[NPMAX];

#include "simulation.impl.h"
