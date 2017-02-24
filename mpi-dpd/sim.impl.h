namespace sim {
static void update_helper_arrays() {
  CC(cudaFuncSetCacheConfig(k_sim::make_texture, cudaFuncCachePreferShared));
  int np = s_pp->S;
  xyzouvwo->resize(2 * np);
  xyzo_half->resize(np);
  if (np)
    k_sim::make_texture<<<(np + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>(
	xyzouvwo->D, xyzo_half->D, (float *)s_pp->D, np);
}

/* set initial velocity of a particle */
static void ic_vel0(float x, float y, float z,
		    float *vx, float *vy, float *vz) {
  *vx = gamma_dot * z; *vy = 0; *vz = 0; /* TODO: works only for one
					    processor */
}

static void ic_vel(Particle* pp, int np) { /* assign particle
					velocity based on position */
  for (int ip = 0; ip < np; ip++) {
    Particle p = pp[ip];
    float x = p.r[0], y = p.r[1], z = p.r[2], vx, vy, vz;
    ic_vel0(x, y, z, &vx, &vy, &vz);
    p.v[0] = vx; p.v[1] = vy; p.v[2] = vz;
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
	  p.r[0] = x; p.r[1] = y; p.r[2] = z;
	  pp.push_back(p);
	}
      }
  fprintf(stderr, "(simulation) generated %d particles\n", pp.size());
  return pp;
}

static std::vector<Particle> _ic() { /* initial conditions for position and
				 velocity */
  std::vector<Particle> pp = _ic_pos();
  ic_vel(&pp.front(), pp.size());
  return pp;
}

static void redistribute() {
  sdstr::pack(s_pp->D, s_pp->S);
  if (rbcs) rdstr::extent(r_pp->D, Cont::ncells);
  sdstr::send();
  if (rbcs) rdstr::pack_sendcount(r_pp->D, Cont::ncells);
  sdstr::bulk(s_pp->S, cells->start, cells->count);
  const int newnp = sdstr::recv_count();
  if (rbcs) {
    Cont::ncells = rdstr::post();
    r_pp->resize(Cont::ncells*Cont::nvertices);
    r_ff->resize(Cont::ncells*Cont::nvertices);
  }
  s_pp0->resize(newnp); s_ff0->resize(newnp);
  xyzouvwo->resize(newnp * 2);
  xyzo_half->resize(newnp);
  sdstr::recv_unpack(s_pp0->D,
		     xyzouvwo->D, xyzo_half->D,
		     newnp, cells->start, cells->count);
  swap(s_pp, s_pp0); swap(s_ff, s_ff0);
  if (rbcs) rdstr::unpack(r_pp->D, Cont::ncells);
}

void remove_bodies_from_wall() {
  if (!rbcs)         return;
  if (!Cont::ncells) return;
  DeviceBuffer<int> marks(Cont::pcount());
  k_wall::fill_keys<<<(Cont::pcount() + 127) / 128, 128>>>
    (r_pp->D, Cont::pcount(), marks.D);

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
  rbc_remove_resize(r_pp, r_ff, &tokill.front(), tokill.size());
  Cont::clear_velocity(r_pp);
}

void create_walls() {
  int nsurvived = 0;
  ExpectedMessageSizes new_sizes;
  wall::init(s_pp->D, s_pp->S,
	    nsurvived, new_sizes); wall_created = true;
  resize2(s_pp, s_ff, nsurvived);
  Cont::clear_velocity(s_pp);
  cells->build(s_pp->D, s_pp->S, NULL, NULL);
  update_helper_arrays();

  // remove cells touching the wall
  remove_bodies_from_wall();
}

void forces_rbc() {
  if (rbcs)
    CudaRBC::forces_nohost(Cont::ncells,
			   (float*)r_pp->D, (float*)r_ff->D);
}

void forces_dpd() {
  DPD::pack(s_pp->D, s_pp->S, cells->start, cells->count);
  DPD::local_interactions(s_pp->D, xyzouvwo->D, xyzo_half->D,
			  s_pp->S, s_ff->D, cells->start,
			  cells->count);
  DPD::post(s_pp->D, s_pp->S);
  DPD::recv();
  DPD::remote_interactions(s_pp->D, s_pp->S, s_ff->D);
}

void clear_forces() {
  Cont::clear_forces(s_ff);
  if (rbcs) Cont::clear_forces(r_ff);
}

void forces_wall() {
  if (rbcs && wall_created) wall::interactions(r_pp->D, Cont::pcount(), r_ff->D);
  if (wall_created)         wall::interactions(s_pp->D, s_pp->S, s_ff->D);
}

void forces_cnt(std::vector<ParticlesWrap> *w_r) {
  if (contactforces) {
    cnt::build_cells(*w_r);
    cnt::bulk(*w_r);
  }
}

void forces_fsi(SolventWrap *w_s, std::vector<ParticlesWrap> *w_r) {
  fsi::bind_solvent(*w_s);
  fsi::bulk(*w_r);
}

void forces() {
  SolventWrap w_s(s_pp->D, s_pp->S, s_ff->D, cells->start, cells->count);
  std::vector<ParticlesWrap> w_r;
  if (rbcs) w_r.push_back(ParticlesWrap(r_pp->D, Cont::pcount(), r_ff->D));

  clear_forces();

  forces_dpd();
  forces_wall();
  forces_rbc();

  forces_cnt(&w_r);
  forces_fsi(&w_s, &w_r);

  rex::bind_solutes(w_r);
  rex::pack_p();
  rex::post_p();
  rex::recv_p();

  rex::halo(); /* fsi::halo(); cnt::halo() */

  rex::post_f();
  rex::recv_f();
}

void dump_init() {
  MC(MPI_Comm_dup(activecomm, &myactivecomm));
  MC(MPI_Comm_dup(Cont::cartcomm, &mycartcomm));
  dump_field = new H5FieldDump (Cont::cartcomm);

  int rank;
  MC(MPI_Comm_rank(myactivecomm, &rank));
  MC(MPI_Barrier(myactivecomm));
}

void dump_final() {
  delete dump_field;
  delete dump_part_solvent;
  CC(cudaEventDestroy(evdownloaded));
}

static void datadump_async(int idtimestep) {
  static int iddatadump = 0;
    CC(cudaEventSynchronize(evdownloaded));

    int n = particles_datadump->S;
    Particle *p = particles_datadump->D;
    Force *a = forces_datadump->D;

    diagnostics(myactivecomm, mycartcomm, p, n, dt, datadump_idtimestep);
    if (hdf5part_dumps) {
      if (!dump_part_solvent && walls && datadump_idtimestep >= wall_creation_stepid) {
	dump_part_solvent =
	    new H5PartDump("solvent-particles.h5part", activecomm, Cont::cartcomm);
      }
      if (dump_part_solvent) dump_part_solvent->dump(p, n);
    }

    if (hdf5field_dumps && (datadump_idtimestep % steps_per_hdf5dump == 0))
      dump_field->dump(activecomm, p, datadump_nsolvent, datadump_idtimestep);

    if (rbcs)
      Cont::rbc_dump(myactivecomm, p + datadump_nsolvent,
		     a + datadump_nsolvent, datadump_nrbcs, iddatadump);
    ++iddatadump;
}

void datadump(const int idtimestep) {
  int n = s_pp->S;
  if (rbcs) n += Cont::pcount();
  particles_datadump->resize(n);
  forces_datadump->resize(n);
#include "sim.hack.h"
  CC(cudaMemcpyAsync(particles_datadump->D, s_pp->D,
			     sizeof(Particle) * s_pp->S,
			     cudaMemcpyDeviceToHost, 0));
  CC(cudaMemcpyAsync(forces_datadump->D, s_ff->D,
			     sizeof(Force) * s_pp->S,
			     cudaMemcpyDeviceToHost, 0));
  int start = s_pp->S;
  if (rbcs) {
    CC(cudaMemcpyAsync(
	particles_datadump->D + start, r_pp->D,
	sizeof(Particle) * Cont::pcount(), cudaMemcpyDeviceToHost, 0));
    CC(cudaMemcpyAsync(
	forces_datadump->D + start, r_ff->D,
	sizeof(Force) * Cont::pcount(), cudaMemcpyDeviceToHost, 0));
    start += Cont::pcount();
  }
  CC(cudaEventRecord(evdownloaded, 0));

  datadump_idtimestep = idtimestep;
  datadump_nsolvent = s_pp->S;
  datadump_nrbcs = rbcs ? Cont::pcount() : 0;
  datadump_async(idtimestep);
}

static void update_and_bounce() {
  Cont::upd_stg2_and_1(s_pp, s_ff, false, driving_force);
  if (rbcs) Cont::upd_stg2_and_1(r_pp, r_ff, true, driving_force);
  if (wall_created) {
    wall::bounce(s_pp->D, s_pp->S);
    if (rbcs) wall::bounce(r_pp->D, Cont::pcount());
  }
}

void init(MPI_Comm cartcomm_, MPI_Comm activecomm_) {
  Cont::cartcomm = cartcomm_; activecomm = activecomm_;
  rdstr::redistribute_rbcs_init(Cont::cartcomm);
  DPD::init(Cont::cartcomm);
  fsi::init(Cont::cartcomm);
  rex::init(Cont::cartcomm);
  cnt::init(Cont::cartcomm);
  cells   = new CellLists(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN);
  particles_datadump     = new StaticHostBuffer<Particle>;
  forces_datadump = new StaticHostBuffer<Force>;

  xyzouvwo    = new StaticDeviceBuffer<float4>;
  xyzo_half = new StaticDeviceBuffer<ushort4>;
  if (rbcs) {
    r_pp = new StaticDeviceBuffer<Particle>;
    r_ff = new StaticDeviceBuffer<Force>;
  }

  wall::trunk = new Logistic::KISS;
  sdstr::redist_part_init(Cont::cartcomm);
  nsteps = (int)(tend / dt);
  MC(MPI_Comm_rank(activecomm, &Cont::rank));

  int dims[3], periods[3]; /* `coords' is global */
  MC(MPI_Cart_get(Cont::cartcomm, 3, dims, periods, Cont::coords));
  Cont::origin = make_float3((0.5 + Cont::coords[0]) * XSIZE_SUBDOMAIN,
			     (0.5 + Cont::coords[1]) * YSIZE_SUBDOMAIN,
			     (0.5 + Cont::coords[2]) * ZSIZE_SUBDOMAIN);
  Cont::globalextent = make_float3(dims[0] * XSIZE_SUBDOMAIN,
				   dims[1] * YSIZE_SUBDOMAIN,
				   dims[2] * ZSIZE_SUBDOMAIN);
  s_pp  = new StaticDeviceBuffer<Particle>;
  s_ff  = new StaticDeviceBuffer<Force>;
  s_pp0 = new StaticDeviceBuffer<Particle>;
  s_ff0 = new StaticDeviceBuffer<Force>;

  vector<Particle> ic = _ic();
  resize2(s_pp, s_ff  , ic.size());
  resize2(s_pp0, s_ff , ic.size());
  CC(cudaMemcpy(s_pp->D, &ic.front(),
			sizeof(Particle) * ic.size(),
			cudaMemcpyHostToDevice));
  cells->build(s_pp->D, s_pp->S, NULL, NULL);
  update_helper_arrays();

  if (rbcs) {
    Cont::rbc_init();
    Cont::setup(r_pp, r_ff, "rbcs-ic.txt");
  }

  CC(cudaEventCreate(&evdownloaded,
			     cudaEventDisableTiming | cudaEventBlockingSync));
  particles_datadump->resize(s_pp->S * 1.5);
  forces_datadump->resize(s_pp->S * 1.5);

  dump_init();
}

void run() {
  if (Cont::rank == 0 && !walls) printf("will take %ld steps\n", nsteps);
  if (!walls && pushtheflow) driving_force = hydrostatic_a;
  int it;
  for (it = 0; it < nsteps; ++it) {
    if (walls && it >= wall_creation_stepid && !wall_created) {
      CC(cudaDeviceSynchronize());
      create_walls();
      if (pushtheflow) driving_force = hydrostatic_a;
      if (Cont::rank == 0)
	fprintf(stderr, "the simulation consists of %ld steps\n", nsteps - it);
    }
    redistribute();
    forces();
    if (it % steps_per_dump == 0) datadump(it);
    update_and_bounce();
  }
}

void close() {
  dump_final();
  sdstr::redist_part_close();

  delete r_pp; delete r_ff;

  cnt::close();
  delete cells;
  rex::close();
  fsi::close();
  DPD::close();
  rdstr::redistribute_rbcs_close();

  delete particles_datadump;
  delete forces_datadump;
  delete xyzouvwo;
  delete xyzo_half;
  delete wall::trunk;
  delete s_pp; delete s_ff;
  delete s_pp0; delete s_ff0;
}
}
