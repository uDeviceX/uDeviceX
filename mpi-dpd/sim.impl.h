namespace sim {
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
  sdstr::pack(s_pp, s_n);
  if (rbcs) rdstr::extent(r_pp, Cont::nc, Cont::nv);
  sdstr::send();
  if (rbcs) rdstr::pack_sendcnt(r_pp, Cont::nc, Cont::nv);
  sdstr::bulk(s_n, cells->start, cells->count);
  s_n = sdstr::recv_count();
  if (rbcs) {
    Cont::nc = rdstr::post(Cont::nv); r_n = Cont::nc * Cont::nv;
  }
  sdstr::recv_unpack(s_pp0, s_zip0, s_zip1, s_n, cells->start, cells->count);
  std::swap(s_pp, s_pp0); std::swap(s_ff, s_ff0);
  if (rbcs) rdstr::unpack(r_pp, Cont::nc, Cont::nv);
}

void remove_bodies_from_wall() {
  if (!rbcs)         return;
  if (!Cont::nc) return;
  DeviceBuffer<int> marks(r_n);
  k_wall::fill_keys<<<(r_n + 127) / 128, 128>>>
    (r_pp, r_n, marks.D);

  std::vector<int> tmp(marks.S);
  CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S, D2H));
  int nbodies = Cont::nc;
  std::vector<int> tokill;
  for (int i = 0; i < nbodies; ++i) {
    bool valid = true;
    for (int j = 0; j < Cont::nv && valid; ++j)
      valid &= 0 == tmp[j + Cont::nv * i];
    if (!valid) tokill.push_back(i);
  }

  Cont::nc = Cont::rbc_remove(r_pp, Cont::nv, &tokill.front(), tokill.size());
  r_n = Cont::nc * Cont::nv;
}

static void update_helper_arrays() {
  CC(cudaFuncSetCacheConfig(k_sim::make_texture, cudaFuncCachePreferShared));
  k_sim::make_texture<<<(s_n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>
    (s_zip0, s_zip1, (float*)s_pp, s_n);
}

void create_walls() {
  CC(cudaDeviceSynchronize());
  s_n = wall::init(s_pp, s_n); /* number of survived particles */
  wall_created = true;

  Cont::clear_velocity(s_pp, s_n);
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();

  // remove cells touching the wall
  remove_bodies_from_wall();
}

void forces_rbc() {
  if (rbcs)
    rbc::forces_nohost(Cont::nc, (float*)r_pp, (float*)r_ff);
}

void forces_dpd() {
  DPD::pack(s_pp, s_n, cells->start, cells->count);
  DPD::local_interactions(s_pp, s_zip0, s_zip1,
			  s_n, s_ff, cells->start,
			  cells->count);
  DPD::post(s_pp, s_n);
  DPD::recv();
  DPD::remote_interactions(s_n, s_ff);
}

void clear_forces() {
  Cont::clear_forces(s_ff, s_n);
  if (rbcs) Cont::clear_forces(r_ff, r_n);
}

void forces_wall() {
  if (rbcs && wall_created) wall::interactions(r_pp, r_n, r_ff);
  if (wall_created)         wall::interactions(s_pp, s_n, s_ff);
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
  SolventWrap w_s(s_pp, s_n, s_ff, cells->start, cells->count);
  std::vector<ParticlesWrap> w_r;
  if (rbcs) w_r.push_back(ParticlesWrap(r_pp, r_n, r_ff));

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
#ifdef GWRP
#include "sim.hack.h"
#endif
}

void dev2hst() { /* device2host  */
  CC(cudaMemcpyAsync(sr_pp, s_pp,
		     sizeof(Particle) * s_n, D2H, 0));
  if (rbcs)
    CC(cudaMemcpyAsync(&sr_pp[s_n], r_pp,
		       sizeof(Particle) * r_n, D2H, 0));
}

void dump_part() {
  if (!hdf5part_dumps) return;
  dev2hst(); /* TODO: do not need `r' */
  int n = s_n + r_n;
  dump_part_solvent->dump(sr_pp, n);
}

void dump_rbcs() {
  if (!rbcs) return;
  static int id = 0;
  dev2hst();  /* TODO: do not need `s' */
  Cont::rbc_dump(&sr_pp[s_n], r_n, id++);
}

void dump_grid() {
  if (!hdf5field_dumps) return;
  dev2hst();  /* TODO: do not need `r' */
  dump_field->dump(sr_pp, s_n);
}

void diag(int it) {
  int n = s_n + r_n; dev2hst();
  diagnostics(sr_pp, n, dt, it);
}

static void update_and_bounce() {
  Cont::upd_stg2_and_1(s_pp, s_ff, s_n, false, driving_force);
  if (rbcs)
    Cont::upd_stg2_and_1(r_pp, r_ff, s_n, true, driving_force);
  if (wall_created) {
    wall::bounce(s_pp, s_n);
    if (rbcs) wall::bounce(r_pp, r_n);
  }
}

void init() {
  rbc::setup();
  rdstr::init();
  DPD::init();
  fsi::init();
  rex::init();
  cnt::init();
  if (hdf5part_dumps)
    dump_part_solvent = new H5PartDump("s.h5part");

  cells   = new CellLists(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN);
  mpDeviceMalloc(&s_zip0); mpDeviceMalloc(&s_zip1);

  if (rbcs) {
      mpDeviceMalloc(&r_pp); mpDeviceMalloc(&r_ff);
  }

  wall::trunk = new Logistic::KISS;
  sdstr::init();
  Cont::origin = make_float3((0.5 + m::coords[0]) * XSIZE_SUBDOMAIN,
			     (0.5 + m::coords[1]) * YSIZE_SUBDOMAIN,
			     (0.5 + m::coords[2]) * ZSIZE_SUBDOMAIN);
  Cont::globalextent = make_float3(m::dims[0] * XSIZE_SUBDOMAIN,
				   m::dims[1] * YSIZE_SUBDOMAIN,
				   m::dims[2] * ZSIZE_SUBDOMAIN);
  mpDeviceMalloc(&s_pp); mpDeviceMalloc(&s_pp0);
  mpDeviceMalloc(&s_ff); mpDeviceMalloc(&s_ff0);
  mpDeviceMalloc(&r_ff); mpDeviceMalloc(&r_ff);

  std::vector<Particle> ic = ic_pos();
  s_n  = ic.size();

  CC(cudaMemcpy(s_pp, &ic.front(), sizeof(Particle) * ic.size(), H2D));
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();

  if (rbcs) {
    Cont::rbc_init();
    Cont::nc = Cont::setup(r_pp, "rbcs-ic.txt");
    r_n = Cont::nc * Cont::nv;
#ifdef GWRP
    iotags_init_file("rbc.dat");
    iotags_domain(0, 0, 0,
		  XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN,
		  periods[0], periods[1], periods[0]);
#endif
  }

  dump_field = new H5FieldDump;
  MC(MPI_Barrier(m::cart));
}

void dumps_diags(int it) {
  if (it % steps_per_dump == 0)     in_out();
  if (it % steps_per_dump == 0)     dump_rbcs();
  if (it % steps_per_dump == 0)     dump_part();
  if (it % steps_per_hdf5dump == 0) dump_grid();
  if (it % steps_per_dump == 0)     diag(it);
}

void run() {
  int nsteps = (int)(tend / dt);
  if (Cont::rank == 0 && !walls) printf("will take %ld steps\n", nsteps);
  if (!walls && pushtheflow) driving_force = hydrostatic_a;
  int it;
  for (it = 0; it < nsteps; ++it) {
    if (walls && it == wall_creation_stepid) {
      create_walls();
      Cont::clear_velocity(r_pp, r_n);
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

  cnt::close();
  delete cells;
  rex::close();
  fsi::close();
  DPD::close();
  rdstr::close();

  CC(cudaFree(s_zip0));
  CC(cudaFree(s_zip1));

  delete wall::trunk;
  CC(cudaFree(r_pp )); CC(cudaFree(r_ff));
  CC(cudaFree(s_pp )); CC(cudaFree(s_ff ));
  CC(cudaFree(s_pp0)); CC(cudaFree(s_ff0));
}
}
