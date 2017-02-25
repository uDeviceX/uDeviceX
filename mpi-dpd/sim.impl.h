namespace sim {
static void update_helper_arrays() {
  CC(cudaFuncSetCacheConfig(k_sim::make_texture, cudaFuncCachePreferShared));
  int np = s_pp->S;
  if (np)
    k_sim::make_texture<<<(np + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>(
	s_zip0, s_zip1, (float *)s_pp->D, np);
}

static std::vector<Particle> ic_pos() { /* generate particle position */
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
	  p.v[0] = 0; p.v[1] = 0; p.v[2] = 0;
	  pp.push_back(p);
	}
      }
  fprintf(stderr, "(simulation) generated %d particles\n", pp.size());
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
  sdstr::recv_unpack(s_pp0->D,
		     s_zip0, s_zip1,
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
  CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S, D2H));
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
  CC(cudaDeviceSynchronize());

  int nsurvived = 0;
  wall::init(s_pp->D, s_pp->S, nsurvived);
  wall_created = true;

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
  DPD::local_interactions(s_pp->D, s_zip0, s_zip1,
			  s_pp->S, s_ff->D, cells->start,
			  cells->count);
  DPD::post(s_pp->D, s_pp->S);
  DPD::recv();
  DPD::remote_interactions(s_pp->S, s_ff->D);
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

void in_out() {
#include "sim.hack.h"
}

void dev2hst() { /* device2host  */
  CC(cudaMemcpyAsync(sr_pp, s_pp->D,
		     sizeof(Particle) * s_pp->S, D2H, 0));
  if (rbcs)
    CC(cudaMemcpyAsync(&sr_pp[s_pp->S], r_pp->D,
		       sizeof(Particle) * Cont::pcount(), D2H, 0));
}

void dump_part() {
  if (!hdf5part_dumps) return;
  dev2hst(); /* TODO: do not need `r' */
  int n = s_pp->S + Cont::pcount();
  dump_part_solvent->dump(sr_pp, n);
}

void dump_rbcs() {
  if (!rbcs) return;
  static int id = 0;
  dev2hst();  /* TODO: do not need `s' */
  Cont::rbc_dump(myactivecomm, &sr_pp[s_pp->S], Cont::pcount(), id++);
}

void dump_grid() {
  if (!hdf5field_dumps) return;
  dev2hst();  /* TODO: do not need `r' */
  dump_field->dump(activecomm, sr_pp, s_pp->S);
}

void diag(int it) {
  int n = s_pp->S + Cont::pcount(); dev2hst();
  diagnostics(myactivecomm, mycartcomm, sr_pp, n, dt, it);
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
  if (hdf5part_dumps)
    dump_part_solvent = new H5PartDump("s.h5part", activecomm, Cont::cartcomm);

  cells   = new CellLists(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN);

  mpDeviceMalloc(&s_zip0);
  mpDeviceMalloc(&s_zip1);

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

  vector<Particle> ic = ic_pos();
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

  MC(MPI_Comm_dup(activecomm, &myactivecomm));
  MC(MPI_Comm_dup(Cont::cartcomm, &mycartcomm));
  dump_field = new H5FieldDump (Cont::cartcomm);

  int rank;
  MC(MPI_Comm_rank(myactivecomm, &rank));
  MC(MPI_Barrier(myactivecomm));
}

void dumps_diags(int it) {
  if (it % steps_per_dump == 0)     in_out();
  if (it % steps_per_dump == 0)     dump_rbcs();
  if (it % steps_per_dump == 0)     dump_part();
  if (it % steps_per_hdf5dump == 0) dump_grid();
  if (it % steps_per_dump == 0)     diag(it);
}

void run() {
  if (Cont::rank == 0 && !walls) printf("will take %ld steps\n", nsteps);
  if (!walls && pushtheflow) driving_force = hydrostatic_a;
  int it;
  for (it = 0; it < nsteps; ++it) {
    if (walls && it == wall_creation_stepid) {
      create_walls();
      if (pushtheflow) driving_force = hydrostatic_a;
    }
    redistribute();
    forces();
    dumps_diags(it);
    update_and_bounce();
  }
}

void close() {
  delete dump_field;
  delete dump_part_solvent;
  sdstr::redist_part_close();

  delete r_pp; delete r_ff;

  cnt::close();
  delete cells;
  rex::close();
  fsi::close();
  DPD::close();
  rdstr::redistribute_rbcs_close();

  CC(cudaFree(s_zip0));
  CC(cudaFree(s_zip1));

  delete wall::trunk;
  delete s_pp; delete s_ff;
  delete s_pp0; delete s_ff0;
}
}
