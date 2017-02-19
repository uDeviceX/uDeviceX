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
  redistribute->pack(particles->pp.D, particles->S, mainstream);

  CC(cudaPeekAtLastError());

  if (rbcscoll)
    redistribute_rbcs->extent(rbcscoll->pp.D, rbcscoll->ncells, mainstream);

  redistribute->send();

  if (rbcscoll)
    redistribute_rbcs->pack_sendcount(rbcscoll->pp.D, rbcscoll->ncells,
				     mainstream);

  redistribute->bulk(particles->S, cells->start, cells->count, mainstream);

  CC(cudaPeekAtLastError());

  const int newnp = redistribute->recv_count(mainstream);

  int nrbcs;
  if (rbcscoll)
    nrbcs = redistribute_rbcs->post();

  if (rbcscoll)
    rbcscoll->rbc_resize(nrbcs);

  newparticles->pa_resize(newnp);
  xyzouvwo->resize(newnp * 2);
  xyzo_half->resize(newnp);

  redistribute->recv_unpack(newparticles->pp.D,
			    xyzouvwo->D, xyzo_half->D,
			    newnp, cells->start, cells->count,
			    mainstream);

  CC(cudaPeekAtLastError());

  swap(particles, newparticles);

  if (rbcscoll)
    redistribute_rbcs->unpack(rbcscoll->pp.D, rbcscoll->ncells, mainstream);

  CC(cudaPeekAtLastError());
}

void sim_remove_bodies_from_wall(CollectionRBC *coll) {
  if (!coll || !coll->ncells) return;
  SimpleDeviceBuffer<int> marks(coll->pcount());

  SolidWallsKernel::fill_keys<<<(coll->pcount() + 127) / 128, 128>>>(
      coll->pp.D, coll->pcount(), marks.D);

  vector<int> tmp(marks.S);
  CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S,
			cudaMemcpyDeviceToHost));

  const int nbodies = coll->ncells;

  std::vector<int> tokill;
  for (int i = 0; i < nbodies; ++i) {
    bool valid = true;

    for (int j = 0; j < nvertices && valid; ++j)
      valid &= 0 == tmp[j + nvertices * i];

    if (!valid)
      tokill.push_back(i);
  }

  coll->remove(&tokill.front(), tokill.size());
  coll->clear_velocity();

  CC(cudaPeekAtLastError());
}

void sim_create_walls() {
  int nsurvived = 0;
  ExpectedMessageSizes new_sizes;
  wall_init(cartcomm, particles->pp.D, particles->S,
	    nsurvived, new_sizes); wall_created = true;
  particles->pa_resize(nsurvived);
  particles->clear_velocity();
  cells->build(particles->pp.D, particles->S, 0, NULL, NULL);
  sim_update_helper_arrays();
  CC(cudaPeekAtLastError());

  // remove cells touching the wall
  sim_remove_bodies_from_wall(rbcscoll);
  H5PartDump sd("survived-particles.h5part", activecomm, cartcomm);
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

  if (rbcscoll)
    wsolutes.push_back(ParticlesWrap(rbcscoll->pp.D, rbcscoll->pcount(), rbcscoll->aa.D));

  fsi->bind_solvent(wsolvent);

  solutex->bind_solutes(wsolutes);

  particles->clear_acc(mainstream);

  if (rbcscoll)
    rbcscoll->clear_acc(mainstream);

  dpd->pack(particles->pp.D, particles->S, cells->start, cells->count,
	   mainstream);

  solutex->pack_p(mainstream);

  CC(cudaPeekAtLastError());

  if (contactforces)
    contact->build_cells(wsolutes, mainstream);

  dpd->local_interactions(particles->pp.D, xyzouvwo->D, xyzo_half->D,
			 particles->S, particles->aa.D, cells->start,
			 cells->count, mainstream);

  dpd->post(particles->pp.D, particles->S, mainstream, downloadstream);

  solutex->post_p(mainstream, downloadstream);

  CC(cudaPeekAtLastError());

  if (rbcscoll && wall_created)
    wall_interactions(rbcscoll->pp.D, rbcscoll->pcount(), rbcscoll->aa.D,
		       mainstream);

  if (wall_created)
    wall_interactions(particles->pp.D, particles->S,
		      particles->aa.D, mainstream);

  CC(cudaPeekAtLastError());

  dpd->recv(mainstream, uploadstream);

  solutex->recv_p(uploadstream);

  solutex->halo(uploadstream, mainstream);

  dpd->remote_interactions(particles->pp.D, particles->S,
			   particles->aa.D, mainstream, uploadstream);

  fsi->bulk(wsolutes, mainstream);

  if (contactforces)
    contact->bulk(wsolutes, mainstream);

  CC(cudaPeekAtLastError());

  if (rbcscoll)
    CudaRBC::forces_nohost(mainstream, rbcscoll->ncells,
			   (float *)rbcscoll->pp.D, (float *)rbcscoll->aa.D);

  CC(cudaPeekAtLastError());

  solutex->post_a();

  solutex->recv_a(mainstream);
  CC(cudaPeekAtLastError());
}

void sim_datadump(const int idtimestep) {
  pthread_mutex_lock(&mutex_datadump);

  while (datadump_pending)
    pthread_cond_wait(&done_datadump, &mutex_datadump);

  int n = particles->S;

  if (rbcscoll)
    n += rbcscoll->pcount();

  particles_datadump->resize(n);
  accelerations_datadump->resize(n);

#include "simulation.hack.h"

  CC(cudaMemcpyAsync(particles_datadump->data, particles->pp.D,
			     sizeof(Particle) * particles->S,
			     cudaMemcpyDeviceToHost, 0));

  CC(cudaMemcpyAsync(accelerations_datadump->data, particles->aa.D,
			     sizeof(Acceleration) * particles->S,
			     cudaMemcpyDeviceToHost, 0));

  int start = particles->S;

  if (rbcscoll) {
    CC(cudaMemcpyAsync(
	particles_datadump->data + start, rbcscoll->pp.D,
	sizeof(Particle) * rbcscoll->pcount(), cudaMemcpyDeviceToHost, 0));
    CC(cudaMemcpyAsync(
	accelerations_datadump->data + start, rbcscoll->aa.D,
	sizeof(Acceleration) * rbcscoll->pcount(), cudaMemcpyDeviceToHost, 0));
    start += rbcscoll->pcount();
  }

  CC(cudaEventRecord(evdownloaded, 0));

  datadump_idtimestep = idtimestep;
  datadump_nsolvent = particles->S;
  datadump_nrbcs = rbcscoll ? rbcscoll->pcount() : 0;
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

  MPI_CHECK(MPI_Comm_dup(activecomm, &myactivecomm));
  MPI_CHECK(MPI_Comm_dup(cartcomm, &mycartcomm));

  H5PartDump dump_part("allparticles.h5part", activecomm, cartcomm),
      *dump_part_solvent = NULL;
  H5FieldDump dump_field(cartcomm);

  MPI_CHECK(MPI_Comm_rank(myactivecomm, &rank));
  MPI_CHECK(MPI_Barrier(myactivecomm));

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
    Particle *p = particles_datadump->data;
    Acceleration *a = accelerations_datadump->data;

    { diagnostics(myactivecomm, mycartcomm, p, n, dt, datadump_idtimestep, a); }

    if (hdf5part_dumps) {
      if (!dump_part_solvent && walls && datadump_idtimestep >= wall_creation_stepid) {
	dump_part.close();

	dump_part_solvent =
	    new H5PartDump("solvent-particles.h5part", activecomm, cartcomm);
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
      if (rbcscoll)
	rbc_dump(myactivecomm, mycartcomm, p + datadump_nsolvent,
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
  particles->upd_stg2_and_1(false, driving_acceleration, mainstream);

  CC(cudaPeekAtLastError());

  if (rbcscoll)
    rbcscoll->upd_stg2_and_1(true, driving_acceleration, mainstream);

  CC(cudaPeekAtLastError());
  if (wall_created) {
    wall_bounce(particles->pp.D, particles->S, mainstream);

    if (rbcscoll)
      wall_bounce(rbcscoll->pp.D, rbcscoll->pcount(), mainstream);
  }

  CC(cudaPeekAtLastError());
}

void sim_init(MPI_Comm cartcomm_, MPI_Comm activecomm_) {
  cartcomm = cartcomm_; activecomm = activecomm_;
  redistribute      = new RedistributeParticles(cartcomm);
  redistribute_rbcs = new RedistributeRBCs(cartcomm);
  dpd     = new ComputeDPD(cartcomm);
  fsi     = new ComputeFSI(cartcomm);
  solutex = new SoluteExchange(cartcomm);
  contact = new ComputeContact(cartcomm);
  cells   = new CellLists(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN);

  particles_datadump     = new PinnedHostBuffer<Particle>;
  accelerations_datadump = new PinnedHostBuffer<Acceleration>;

  xyzouvwo    = new SimpleDeviceBuffer<float4 >;
  xyzo_half = new SimpleDeviceBuffer<ushort4>;
  particles_pingpong[0] = new ParticleArray();
  particles_pingpong[1] = new ParticleArray();
  nsteps = (int)(tend / dt);

  MPI_CHECK(MPI_Comm_size(activecomm, &nranks));
  MPI_CHECK(MPI_Comm_rank(activecomm, &rank));

  solutex->attach_halocomputation(fsi);

  if (contactforces) solutex->attach_halocomputation(contact);

  int dims[3], periods[3]; /* `coords' is global */
  MPI_CHECK(MPI_Cart_get(cartcomm, 3, dims, periods, coords));
  origin = make_float3((0.5 + coords[0]) * XSIZE_SUBDOMAIN,
		       (0.5 + coords[1]) * YSIZE_SUBDOMAIN,
		       (0.5 + coords[2]) * ZSIZE_SUBDOMAIN);
  globalextent = make_float3(dims[0] * XSIZE_SUBDOMAIN,
			     dims[1] * YSIZE_SUBDOMAIN,
			     dims[2] * ZSIZE_SUBDOMAIN);
  particles =     particles_pingpong[0];
  newparticles =  particles_pingpong[1];

  vector<Particle> ic = _ic();

  for (int c = 0; c < 2; ++c)
    particles_pingpong[c]->pa_resize(ic.size());


  CC(cudaMemcpy(particles->pp.D, &ic.front(),
			sizeof(Particle) * ic.size(),
			cudaMemcpyHostToDevice));
  cells->build(particles->pp.D, particles->S, 0, NULL, NULL);
  sim_update_helper_arrays();

  CC(cudaStreamCreate(&mainstream));
  CC(cudaStreamCreate(&uploadstream));
  CC(cudaStreamCreate(&downloadstream));

  if (rbcs) {
    rbcscoll = new CollectionRBC(cartcomm);
    rbcscoll->setup("rbcs-ic.txt");
  }

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
	ParticlesWrap(rbcscoll->pp.D, rbcscoll->pcount(), rbcscoll->aa.D));

  fsi->bind_solvent(wsolvent);
  solutex->bind_solutes(wsolutes);
  particles->clear_acc(mainstream);

  if (rbcscoll)
    rbcscoll->clear_acc(mainstream);

  solutex->pack_p(mainstream);
  dpd->pack(particles->pp.D, particles->S, cells->start, cells->count,
	   mainstream);

  dpd->local_interactions(particles->pp.D, xyzouvwo->D, xyzo_half->D,
			 particles->S, particles->aa.D, cells->start,
			 cells->count, mainstream);
  if (contactforces) contact->build_cells(wsolutes, mainstream);
  solutex->post_p(mainstream, downloadstream);
  dpd->post(particles->pp.D, particles->S, mainstream, downloadstream);
  CC(cudaPeekAtLastError());

  if (wall_created)
    wall_interactions(particles->pp.D, particles->S,
		       particles->aa.D,
		       mainstream);

  CC(cudaPeekAtLastError());
  dpd->recv(mainstream, uploadstream);
  solutex->recv_p(uploadstream);
  solutex->halo(uploadstream, mainstream);
  dpd->remote_interactions(particles->pp.D, particles->S,
			  particles->aa.D, mainstream, uploadstream);

  fsi->bulk(wsolutes, mainstream);

  if (contactforces) contact->bulk(wsolutes, mainstream);
  CC(cudaPeekAtLastError());

  if (rbcscoll)
    CudaRBC::forces_nohost(mainstream, rbcscoll->ncells,
			   (float *)rbcscoll->pp.D, (float *)rbcscoll->aa.D);
  CC(cudaPeekAtLastError());
  solutex->post_a();
  particles->upd_stg2_and_1(false, driving_acceleration, mainstream);
  if (wall_created) wall_bounce(particles->pp.D, particles->S, mainstream);
  CC(cudaPeekAtLastError());
  redistribute->pack(particles->pp.D, particles->S, mainstream);
  redistribute->send();
  redistribute->bulk(particles->S, cells->start, cells->count, mainstream);
  CC(cudaPeekAtLastError());

  if (rbcscoll && wall_created)
    wall_interactions(rbcscoll->pp.D, rbcscoll->pcount(), rbcscoll->aa.D,
		       mainstream);
  CC(cudaPeekAtLastError());
  solutex->recv_a(mainstream);
  if (rbcscoll) rbcscoll->upd_stg2_and_1(true, driving_acceleration, mainstream);
  int newnp = redistribute->recv_count(mainstream);
  CC(cudaPeekAtLastError());
  if (rbcscoll) {
    redistribute_rbcs->extent(rbcscoll->pp.D, rbcscoll->ncells, mainstream);
    redistribute_rbcs->pack_sendcount(rbcscoll->pp.D, rbcscoll->ncells, mainstream);
  }
  newparticles->pa_resize(newnp);
  xyzouvwo->resize(newnp * 2);
  xyzo_half->resize(newnp);
  redistribute->recv_unpack(newparticles->pp.D, xyzouvwo->D,
			   xyzo_half->D, newnp, cells->start, cells->count,
			   mainstream);
  CC(cudaPeekAtLastError());
  swap(particles, newparticles);
  int nrbcs;
  if (rbcscoll) nrbcs = redistribute_rbcs->post();

  if (rbcscoll) rbcscoll->rbc_resize(nrbcs);
  CC(cudaPeekAtLastError());
  if (rbcscoll)
    redistribute_rbcs->unpack(rbcscoll->pp.D, rbcscoll->ncells, mainstream);
  CC(cudaPeekAtLastError());
}

void sim_run() {
  if (rank == 0 && !walls) printf("simulation consists of %ll steps\n", nsteps);
  sim_redistribute();
  sim_forces();
  if (!walls && pushtheflow) driving_acceleration = hydrostatic_a;
  particles->upd_stg1(false, driving_acceleration, mainstream);
  if (rbcscoll) rbcscoll->upd_stg1(true, driving_acceleration, mainstream);

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
      if (rank == 0)
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

  delete rbcscoll;
  delete contact;
  delete cells;
  delete solutex;
  delete fsi;
  delete dpd;
  delete redistribute_rbcs;
  delete redistribute;

  delete particles_datadump;
  delete accelerations_datadump;

  delete xyzouvwo;
  delete xyzo_half;

  delete particles_pingpong[0];
  delete particles_pingpong[1];
}
