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
#include "containers.h"
#include "solvent-exchange.h"
#include "dpd.h"
#include "wall.h"
#include "solute-exchange.h"
#include "fsi.h"
#include "contact.h"
#include "redistribute-particles.h"
#include "redistribute-rbcs.h"
#include "io.h"
#include "simulation.h"
#include "dpd-forces.h"
#include "last_bit_float.h"
#include "geom-wrapper.h"

ParticleArray particles_pingpong[2];
ParticleArray * particles, * newparticles;
SimpleDeviceBuffer<float4> xyzouvwo;
SimpleDeviceBuffer<ushort4> xyzo_half;

CellLists* cells;
CollectionRBC * rbcscoll;

RedistributeParticles* redistribute;
RedistributeRBCs* redistribute_rbcs;

ComputeDPD* dpd;
SoluteExchange* solutex;
ComputeFSI* fsi;
ComputeContact* contact;

ComputeWall * wall;
bool sim_is_done;

MPI_Comm activecomm, cartcomm;
cudaStream_t mainstream, uploadstream, downloadstream;

size_t nsteps;
float driving_acceleration;
int nranks, rank;

pthread_t thread_datadump;
pthread_mutex_t mutex_datadump;
pthread_cond_t request_datadump, done_datadump;
bool datadump_pending;
int datadump_idtimestep, datadump_nsolvent, datadump_nrbcs;
bool async_thread_initialized;

PinnedHostBuffer<Particle> particles_datadump;
PinnedHostBuffer<Acceleration> accelerations_datadump;

cudaEvent_t evdownloaded;

#define NPMAX 5000000 /* TODO: */
float rbc_xx[NPMAX], rbc_yy[NPMAX], rbc_zz[NPMAX];
float sol_xx[NPMAX], sol_yy[NPMAX], sol_zz[NPMAX];
int iotags[NPMAX];

#include "simulation.impl.h"
