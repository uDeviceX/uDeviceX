static __global__ void make_texture(float4 *__restrict xyzouvwoo,
			     ushort4 *__restrict xyzo_half,
			     const float *__restrict xyzuvw, const uint n) {
  extern __shared__ volatile float smem[];
  const uint warpid = threadIdx.x / 32;
  const uint lane = threadIdx.x % 32;

  const uint i = (blockIdx.x * blockDim.x + threadIdx.x) & 0xFFFFFFE0U;

  const float2 *base = (float2 *)(xyzuvw + i * 6);
#pragma unroll 3
  for (uint j = lane; j < 96; j += 32) {
    float2 u = base[j];
    // NVCC bug: no operator = between volatile float2 and float2
    asm volatile("st.volatile.shared.v2.f32 [%0], {%1, %2};"
		 :
		 : "r"((warpid * 96 + j) * 8), "f"(u.x), "f"(u.y)
		 : "memory");
  }
  // SMEM: XYZUVW XYZUVW ...
  uint pid = lane / 2;
  const uint x_or_v = (lane % 2) * 3;
  xyzouvwoo[i * 2 + lane] =
      make_float4(smem[warpid * 192 + pid * 6 + x_or_v + 0],
		  smem[warpid * 192 + pid * 6 + x_or_v + 1],
		  smem[warpid * 192 + pid * 6 + x_or_v + 2], 0);
  pid += 16;
  xyzouvwoo[i * 2 + lane + 32] =
      make_float4(smem[warpid * 192 + pid * 6 + x_or_v + 0],
		  smem[warpid * 192 + pid * 6 + x_or_v + 1],
		  smem[warpid * 192 + pid * 6 + x_or_v + 2], 0);

  xyzo_half[i + lane] =
      make_ushort4(__float2half_rn(smem[warpid * 192 + lane * 6 + 0]),
		   __float2half_rn(smem[warpid * 192 + lane * 6 + 1]),
		   __float2half_rn(smem[warpid * 192 + lane * 6 + 2]), 0);
}

static void sim_update_helper_arrays() {
  CC(cudaFuncSetCacheConfig(make_texture, cudaFuncCachePreferShared));

  const int np = particles->S;

  xyzouvwo->resize(2 * np);
  xyzo_half->resize(np);

  if (np)
    make_texture<<<(np + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>(
	xyzouvwo->D, xyzo_half->D, (float *)particles->pp.D, np);

  CC(cudaPeekAtLastError());
}

/* set initial velocity of a particle */
static void sim_ic_vel0(float x, float y, float z, float *vx, float *vy, float *vz) {
  *vx = gamma_dot * z; *vy = 0; *vz = 0; /* TODO: works only for one
					    processor */
}

static void sim_ic_vel(Particle* pp, int np) { /* assign particle velocity based
					on position */
  for (int ip = 0; ip < np; ip++) {
    Particle p = pp[ip];
    float x = p.x[0], y = p.x[1], z = p.x[2], vx, vy, vz;
    sim_ic_vel0(x, y, z, &vx, &vy, &vz);
    p.u[0] = vx; p.u[1] = vy; p.u[2] = vz;
  }
}

static std::vector<Particle> _ic_pos() { /* generate particle position */
  srand48(0);
  std::vector<Particle> pp;
  int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
  int iz, iy, ix, l, nd = numberdensity;
  float x, y, z, dr = 0.99;
  for (iz = 0; iz < L[2]; iz++)
    for (iy = 0; iy < L[1]; iy++)
      for (ix = 0; ix < L[0]; ix++) {
	/* edge of a cell */
	int xlo = -L[0]/2 + ix, ylo = -L[1]/2 + iy, zlo = -L[2]/2 + iz;
	for (l = 0; l < nd; l++) {
	  Particle p = Particle();
	  x = xlo + dr * drand48(), y = ylo + dr * drand48(), z = zlo + dr * drand48();
	  p.x[0] = x; p.x[1] = y; p.x[2] = z;
	  pp.push_back(p);
	}
      }
  fprintf(stderr, "(simulation) generated %d particles\n", pp.size());
  return pp;
}

static std::vector<Particle> _ic() { /* initial conditions for position and
				 velocity */
  std::vector<Particle> pp = _ic_pos();
  sim_ic_vel(&pp.front(), pp.size());
  return pp;
}

static void sim_redistribute() {
  RedistPart::pack(particles->pp.D, particles->S, mainstream);

  CC(cudaPeekAtLastError());

  if (rbcscoll)
    RedistRBC::extent(rbcscoll->pp.D, Cont::ncells, mainstream);

  RedistPart::send();

  if (rbcscoll)
    RedistRBC::pack_sendcount(rbcscoll->pp.D, Cont::ncells, mainstream);

  RedistPart::bulk(particles->S, cells->start, cells->count, mainstream);

  CC(cudaPeekAtLastError());

  const int newnp = RedistPart::recv_count(mainstream);

  int nrbcs;
  if (rbcs) {
    nrbcs = RedistRBC::post();
    Cont::rbc_resize(rbcscoll, nrbcs);
  }

  Cont::pa_resize(newparticles, newnp);
  xyzouvwo->resize(newnp * 2);
  xyzo_half->resize(newnp);

  RedistPart::recv_unpack(newparticles->pp.D,
			  xyzouvwo->D, xyzo_half->D,
			  newnp, cells->start, cells->count,
			  mainstream);

  CC(cudaPeekAtLastError());

  swap(particles, newparticles);

  if (rbcs)
    RedistRBC::unpack(rbcscoll->pp.D, Cont::ncells, mainstream);

  CC(cudaPeekAtLastError());
}

void sim_remove_bodies_from_wall(ParticleArray *coll) {
  if (!coll || !Cont::ncells) return;
  DeviceBuffer<int> marks(Cont::pcount());

  WallKernels::fill_keys<<<(Cont::pcount() + 127) / 128, 128>>>
    (coll->pp.D, Cont::pcount(), marks.D);

  vector<int> tmp(marks.S);
  CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S, cudaMemcpyDeviceToHost));
  
  int nbodies = Cont::ncells;
  std::vector<int> tokill;
  for (int i = 0; i < nbodies; ++i) {
    bool valid = true;

    for (int j = 0; j < Cont::nvertices && valid; ++j)
      valid &= 0 == tmp[j + Cont::nvertices * i];

    if (!valid) tokill.push_back(i);
  }

  Cont::remove(coll, &tokill.front(), tokill.size());
  Cont::clear_velocity(coll);

  CC(cudaPeekAtLastError());
}

void sim_create_walls() {
  int nsurvived = 0;
  ExpectedMessageSizes new_sizes;
  Wall::init(particles->pp.D, particles->S,
	    nsurvived, new_sizes); wall_created = true;
  Cont::pa_resize(particles, nsurvived);
  Cont::clear_velocity(particles);
  cells->build(particles->pp.D, particles->S, 0, NULL, NULL);
  sim_update_helper_arrays();
  CC(cudaPeekAtLastError());

  // remove cells touching the wall
  sim_remove_bodies_from_wall(rbcscoll);
  H5PartDump sd("survived-particles.h5part", activecomm, Cont::cartcomm);
  Particle *pp = new Particle[particles->S];
  CC(cudaMemcpy(pp, particles->pp.D,
			sizeof(Particle) * particles->S,
			cudaMemcpyDeviceToHost));
  sd.dump(pp, particles->S);
  delete[] pp;
}

void sim_forces() {
  SolventWrap wsolvent(particles->pp.D, particles->S,
		       particles->aa.D, cells->start, cells->count);

  std::vector<ParticlesWrap> wsolutes;

  if (rbcs)
    wsolutes.push_back(ParticlesWrap(rbcscoll->pp.D, Cont::pcount(), rbcscoll->aa.D));

  FSI::bind_solvent(wsolvent);

  SolEx::bind_solutes(wsolutes);

  Cont::clear_acc(particles, mainstream);

  if (rbcs) Cont::clear_acc(rbcscoll, mainstream);

  DPD::pack(particles->pp.D, particles->S, cells->start, cells->count,
	    mainstream);

  SolEx::pack_p(mainstream);

  CC(cudaPeekAtLastError());

  if (contactforces)
    Contact::build_cells(wsolutes, mainstream);

  DPD::local_interactions(particles->pp.D, xyzouvwo->D, xyzo_half->D,
			 particles->S, particles->aa.D, cells->start,
			 cells->count, mainstream);

  DPD::post(particles->pp.D, particles->S, mainstream, downloadstream);

  SolEx::post_p(mainstream, downloadstream);

  CC(cudaPeekAtLastError());

  if (rbcs && wall_created)
    Wall::interactions(rbcscoll->pp.D, Cont::pcount(), rbcscoll->aa.D, mainstream);

  if (wall_created)
    Wall::interactions(particles->pp.D, particles->S,
		      particles->aa.D, mainstream);

  CC(cudaPeekAtLastError());

  DPD::recv(mainstream, uploadstream);

  SolEx::recv_p(uploadstream);
  SolEx::halo(uploadstream, mainstream);

  DPD::remote_interactions(particles->pp.D, particles->S,
			   particles->aa.D, mainstream, uploadstream);

  FSI::bulk(wsolutes, mainstream);

  if (contactforces)
    Contact::bulk(wsolutes, mainstream);

  CC(cudaPeekAtLastError());

  if (rbcs)
    CudaRBC::forces_nohost(mainstream, Cont::ncells,
			   (float *)rbcscoll->pp.D, (float *)rbcscoll->aa.D);

  CC(cudaPeekAtLastError());

  SolEx::post_a();

  SolEx::recv_a(mainstream);
  CC(cudaPeekAtLastError());
}

void sim_datadump(const int idtimestep) {
  pthread_mutex_lock(&mutex_datadump);

  while (datadump_pending)
    pthread_cond_wait(&done_datadump, &mutex_datadump);

  int n = particles->S;
  if (rbcs) n += Cont::pcount();

  particles_datadump->resize(n);
  accelerations_datadump->resize(n);

#include "simulation.hack.h"

  CC(cudaMemcpyAsync(particles_datadump->D, particles->pp.D,
			     sizeof(Particle) * particles->S,
			     cudaMemcpyDeviceToHost, 0));

  CC(cudaMemcpyAsync(accelerations_datadump->D, particles->aa.D,
			     sizeof(Acceleration) * particles->S,
			     cudaMemcpyDeviceToHost, 0));

  int start = particles->S;

  if (rbcs) {
    CC(cudaMemcpyAsync(
	particles_datadump->D + start, rbcscoll->pp.D,
	sizeof(Particle) * Cont::pcount(), cudaMemcpyDeviceToHost, 0));
    CC(cudaMemcpyAsync(
	accelerations_datadump->D + start, rbcscoll->aa.D,
	sizeof(Acceleration) * Cont::pcount(), cudaMemcpyDeviceToHost, 0));
    start += Cont::pcount();
  }

  CC(cudaEventRecord(evdownloaded, 0));

  datadump_idtimestep = idtimestep;
  datadump_nsolvent = particles->S;
  datadump_nrbcs = rbcs ? Cont::pcount() : 0;
  datadump_pending = true;

  pthread_cond_signal(&request_datadump);
#if defined(_SYNC_DUMPS_)
  while (datadump_pending)
    pthread_cond_wait(&done_datadump, &mutex_datadump);
#endif
  pthread_mutex_unlock(&mutex_datadump);
}

static void sim_datadump_async() {
  int iddatadump = 0, rank;
  int curr_idtimestep = -1;

  MPI_Comm myactivecomm, mycartcomm;

  MC(MPI_Comm_dup(activecomm, &myactivecomm));
  MC(MPI_Comm_dup(Cont::cartcomm, &mycartcomm));

  H5PartDump dump_part("allparticles.h5part", activecomm, Cont::cartcomm),
      *dump_part_solvent = NULL;
  H5FieldDump dump_field(Cont::cartcomm);

  MC(MPI_Comm_rank(myactivecomm, &rank));
  MC(MPI_Barrier(myactivecomm));

  while (true) {
    pthread_mutex_lock(&mutex_datadump);
    async_thread_initialized = 1;

    while (!datadump_pending)
      pthread_cond_wait(&request_datadump, &mutex_datadump);

    pthread_mutex_unlock(&mutex_datadump);

    if (curr_idtimestep == datadump_idtimestep)
      if (sim_is_done)
	break;

    CC(cudaEventSynchronize(evdownloaded));

    int n = particles_datadump->size;
    Particle *p = particles_datadump->D;
    Acceleration *a = accelerations_datadump->D;

    { diagnostics(myactivecomm, mycartcomm, p, n, dt, datadump_idtimestep, a); }

    if (hdf5part_dumps) {
      if (!dump_part_solvent && walls && datadump_idtimestep >= wall_creation_stepid) {
	dump_part.close();

	dump_part_solvent =
	    new H5PartDump("solvent-particles.h5part", activecomm, Cont::cartcomm);
      }

      if (dump_part_solvent)
	dump_part_solvent->dump(p, n);
      else
	dump_part.dump(p, n);
    }

    if (hdf5field_dumps && (datadump_idtimestep % steps_per_hdf5dump == 0)) {
      dump_field.dump(activecomm, p, datadump_nsolvent, datadump_idtimestep);
    }

    /* LINA: this is to not to dump the beginning
       if (datadump_idtimestep >= 600/dt) */
    {
      if (rbcs)
	Cont::rbc_dump(myactivecomm, p + datadump_nsolvent,
		 a + datadump_nsolvent, datadump_nrbcs, iddatadump);
    }

    curr_idtimestep = datadump_idtimestep;

    pthread_mutex_lock(&mutex_datadump);

    if (sim_is_done) {
      pthread_mutex_unlock(&mutex_datadump);
      break;
    }

    datadump_pending = false;

    pthread_cond_signal(&done_datadump);

    pthread_mutex_unlock(&mutex_datadump);

    ++iddatadump;
  }

  if (dump_part_solvent)
    delete dump_part_solvent;

  CC(cudaEventDestroy(evdownloaded));
}

static void * datadump_trampoline(void*) { sim_datadump_async(); return NULL; }

static void sim_update_and_bounce() {
  Cont::upd_stg2_and_1(particles, false, driving_acceleration, mainstream);

  CC(cudaPeekAtLastError());

  if (rbcs)
    Cont::upd_stg2_and_1(rbcscoll, true, driving_acceleration, mainstream);

  CC(cudaPeekAtLastError());
  if (wall_created) {
    Wall::bounce(particles->pp.D, particles->S, mainstream);

    if (rbcs)
      Wall::bounce(rbcscoll->pp.D, Cont::pcount(), mainstream);
  }

  CC(cudaPeekAtLastError());
}

void sim_init(MPI_Comm cartcomm_, MPI_Comm activecomm_) {
  Cont::cartcomm = cartcomm_; activecomm = activecomm_;
  RedistRBC::redistribute_rbcs_init(Cont::cartcomm);
  DPD::init(Cont::cartcomm);
  FSI::init(Cont::cartcomm);
  SolEx::init(Cont::cartcomm);
  Contact::init(Cont::cartcomm);
  cells   = new CellLists(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN);

  particles_datadump     = new PinnedHostBuffer<Particle>;
  accelerations_datadump = new PinnedHostBuffer<Acceleration>;

  xyzouvwo    = new DeviceBuffer<float4 >;
  xyzo_half = new DeviceBuffer<ushort4>;
  particles_pingpong[0] = new ParticleArray();
  particles_pingpong[1] = new ParticleArray();
  if (rbcs) rbcscoll    = new ParticleArray();

  Wall::trunk = new Logistic::KISS;

  RedistPart::redist_part_init(Cont::cartcomm);
  
  nsteps = (int)(tend / dt);

  MC(MPI_Comm_size(activecomm, &Cont::nranks));
  MC(MPI_Comm_rank(activecomm, &Cont::rank));

  int dims[3], periods[3]; /* `coords' is global */
  MC(MPI_Cart_get(Cont::cartcomm, 3, dims, periods, Cont::coords));
  Cont::origin = make_float3((0.5 + Cont::coords[0]) * XSIZE_SUBDOMAIN,
			     (0.5 + Cont::coords[1]) * YSIZE_SUBDOMAIN,
			     (0.5 + Cont::coords[2]) * ZSIZE_SUBDOMAIN);
  Cont::globalextent = make_float3(dims[0] * XSIZE_SUBDOMAIN,
				   dims[1] * YSIZE_SUBDOMAIN,
				   dims[2] * ZSIZE_SUBDOMAIN);
  particles =     particles_pingpong[0];
  newparticles =  particles_pingpong[1];

  vector<Particle> ic = _ic();
  for (int c = 0; c < 2; ++c) Cont::pa_resize(particles_pingpong[c], ic.size());
  
  CC(cudaMemcpy(particles->pp.D, &ic.front(),
			sizeof(Particle) * ic.size(),
			cudaMemcpyHostToDevice));
  cells->build(particles->pp.D, particles->S, 0, NULL, NULL);
  sim_update_helper_arrays();

  CC(cudaStreamCreate(&mainstream));
  CC(cudaStreamCreate(&uploadstream));
  CC(cudaStreamCreate(&downloadstream));

  if (rbcs) {Cont::rbc_init(); Cont::setup(rbcscoll, "rbcs-ic.txt");}

  // setting up the asynchronous data dumps
  CC(cudaEventCreate(&evdownloaded,
			     cudaEventDisableTiming | cudaEventBlockingSync));

  particles_datadump->resize(particles->S * 1.5);
  accelerations_datadump->resize(particles->S * 1.5);

  int rcode = pthread_mutex_init(&mutex_datadump, NULL);
  rcode |= pthread_cond_init(&done_datadump, NULL);
  rcode |= pthread_cond_init(&request_datadump, NULL);
  async_thread_initialized = 0;

  rcode |= pthread_create(&thread_datadump, NULL, &datadump_trampoline, NULL);

  while (true) {
    pthread_mutex_lock(&mutex_datadump);
    int done = async_thread_initialized;
    pthread_mutex_unlock(&mutex_datadump);
    if (done) break;
  }
  if (rcode) {printf("ERROR; return code from pthread_create() is %d\n", rcode); exit(-1);}
}

static void sim_lockstep() {
  SolventWrap wsolvent(particles->pp.D, particles->S,
		       particles->aa.D, cells->start, cells->count);
  std::vector<ParticlesWrap> wsolutes;

  if (rbcscoll)
    wsolutes.push_back(
	ParticlesWrap(rbcscoll->pp.D, Cont::pcount(), rbcscoll->aa.D));

  FSI::bind_solvent(wsolvent);
  SolEx::bind_solutes(wsolutes);
  Cont::clear_acc(particles, mainstream);

  if (rbcscoll) Cont::clear_acc(rbcscoll, mainstream);

  SolEx::pack_p(mainstream);
  DPD::pack(particles->pp.D, particles->S, cells->start, cells->count,
	   mainstream);

  DPD::local_interactions(particles->pp.D, xyzouvwo->D, xyzo_half->D,
			 particles->S, particles->aa.D, cells->start,
			 cells->count, mainstream);
  if (contactforces) Contact::build_cells(wsolutes, mainstream);
  SolEx::post_p(mainstream, downloadstream);
  DPD::post(particles->pp.D, particles->S, mainstream, downloadstream);
  CC(cudaPeekAtLastError());

  if (wall_created)
    Wall::interactions(particles->pp.D, particles->S,
		       particles->aa.D,
		       mainstream);

  CC(cudaPeekAtLastError());
  DPD::recv(mainstream, uploadstream);
  SolEx::recv_p(uploadstream);
  SolEx::halo(uploadstream, mainstream);
  DPD::remote_interactions(particles->pp.D, particles->S,
			  particles->aa.D, mainstream, uploadstream);
  FSI::bulk(wsolutes, mainstream);

  if (contactforces) Contact::bulk(wsolutes, mainstream);
  CC(cudaPeekAtLastError());

  if (rbcscoll)
    CudaRBC::forces_nohost(mainstream, Cont::ncells,
			   (float *)rbcscoll->pp.D, (float *)rbcscoll->aa.D);
  CC(cudaPeekAtLastError());
  SolEx::post_a();
  Cont::upd_stg2_and_1(particles, false, driving_acceleration, mainstream);
  if (wall_created) Wall::bounce(particles->pp.D, particles->S, mainstream);
  CC(cudaPeekAtLastError());
  RedistPart::pack(particles->pp.D, particles->S, mainstream);
  RedistPart::send();
  RedistPart::bulk(particles->S, cells->start, cells->count, mainstream);
  CC(cudaPeekAtLastError());

  if (rbcscoll && wall_created)
    Wall::interactions(rbcscoll->pp.D, Cont::pcount(), rbcscoll->aa.D,
		       mainstream);
  CC(cudaPeekAtLastError());
  SolEx::recv_a(mainstream);
  if (rbcscoll) Cont::upd_stg2_and_1(rbcscoll, true, driving_acceleration, mainstream);
  int newnp = RedistPart::recv_count(mainstream);
  CC(cudaPeekAtLastError());
  if (rbcscoll) {
    RedistRBC::extent(rbcscoll->pp.D, Cont::ncells, mainstream);
    RedistRBC::pack_sendcount(rbcscoll->pp.D, Cont::ncells, mainstream);
  }
  Cont::pa_resize(newparticles, newnp);
  xyzouvwo->resize(newnp * 2);
  xyzo_half->resize(newnp);
  RedistPart::recv_unpack(newparticles->pp.D, xyzouvwo->D,
			  xyzo_half->D, newnp, cells->start, cells->count,
			  mainstream);
  CC(cudaPeekAtLastError());
  swap(particles, newparticles);
  int nrbcs;
  if (rbcscoll) nrbcs = RedistRBC::post();

  if (rbcscoll) Cont::rbc_resize(rbcscoll, nrbcs);
  CC(cudaPeekAtLastError());
  if (rbcscoll)
    RedistRBC::unpack(rbcscoll->pp.D, Cont::ncells, mainstream);
  CC(cudaPeekAtLastError());
}

void sim_run() {
  if (Cont::rank == 0 && !walls) printf("simulation consists of %ld steps\n", nsteps);
  sim_redistribute();
  sim_forces();
  if (!walls && pushtheflow) driving_acceleration = hydrostatic_a;
  Cont::upd_stg1(particles, false, driving_acceleration, mainstream);
  if (rbcscoll) Cont::upd_stg1(rbcscoll, true, driving_acceleration, mainstream);

  int it;
  for (it = 0; it < nsteps; ++it) {
    sim_redistribute();
    while (true) {
      const bool lockstep_OK =
	  !(walls && it >= wall_creation_stepid && !wall_created) &&
	  !(it % steps_per_dump == 0) && !(it + 1 == nsteps);
      if (!lockstep_OK) break;
      sim_lockstep();
      ++it;
    }
    if (walls && it >= wall_creation_stepid && !wall_created) {
      CC(cudaDeviceSynchronize());
      sim_create_walls();
      sim_redistribute();
      if (pushtheflow) driving_acceleration = hydrostatic_a;
      if (Cont::rank == 0)
	fprintf(stderr, "the simulation consists of %ld steps\n", nsteps - it);
    }
    sim_forces();
    if (it % steps_per_dump == 0) sim_datadump(it);
    sim_update_and_bounce();
  }
  sim_is_done = true;
}

void sim_close() {
  pthread_mutex_lock(&mutex_datadump);
  datadump_pending = true;
  pthread_cond_signal(&request_datadump);
  pthread_mutex_unlock(&mutex_datadump);
  pthread_join(thread_datadump, NULL);

  CC(cudaStreamDestroy(mainstream));
  CC(cudaStreamDestroy(uploadstream));
  CC(cudaStreamDestroy(downloadstream));

  RedistPart::redist_part_close();

  if (rbcs) delete rbcscoll;

  Contact::close();
  delete cells;
  SolEx::close();
  FSI::close();
  DPD::close();
  if (rbcs) RedistRBC::redistribute_rbcs_close();

  delete particles_datadump;
  delete accelerations_datadump;

  delete xyzouvwo;
  delete xyzo_half;

  delete particles_pingpong[0];
  delete particles_pingpong[1];

  delete Wall::trunk;
}
