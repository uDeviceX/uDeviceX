namespace sim {

static void distr_s() {
  sdstr::pack(s_pp, s_n);
  sdstr::send();
  sdstr::bulk(s_n, cells->start, cells->count);
  s_n = sdstr::recv_count();
  sdstr::recv_unpack(s_pp0, s_zip0, s_zip1, s_n, cells->start, cells->count);
  std::swap(s_pp, s_pp0);
}

void remove_bodies_from_wall() {
  if (!rbcs) return;
  if (!r_nc) return;
  DeviceBuffer<int> marks(r_n);
  k_sdf::fill_keys<<<k_cnf(r_n)>>>(r_pp, r_n, marks.D);

  std::vector<int> tmp(marks.S);
  CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S, D2H));
  std::vector<int> tokill;
  for (int i = 0; i < r_nc; ++i) {
    bool valid = true;
    for (int j = 0; j < r_nv && valid; ++j)
      valid &= 0 == tmp[j + r_nv * i];
    if (!valid) tokill.push_back(i);
  }

  r_nc = Cont::rbc_remove(r_pp, r_nv, r_nc, &tokill.front(), tokill.size());
  r_n = r_nc * r_nv;
}

static void update_helper_arrays() {
  CC(cudaFuncSetCacheConfig(k_sim::make_texture, cudaFuncCachePreferShared));
  k_sim::make_texture<<<(s_n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>
    (s_zip0, s_zip1, (float*)s_pp, s_n);
}

void create_walls() {
  dSync();
  sdf::init();
  s_n = wall::init(s_pp, s_n); /* number of survived particles */
  wall_created = true;

  k_sim::clear_velocity<<<k_cnf(s_n)>>>(s_pp, s_n);
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();
  remove_bodies_from_wall();
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

void clear_forces(Force* ff, int n) {
  CC(cudaMemsetAsync(ff, 0, sizeof(Force) * n));
}

void forces_wall() {
  if (rbcs && wall_created) wall::interactions(r_pp, r_n, r_ff);
  if (wall_created)         wall::interactions(s_pp, s_n, s_ff);
}

void forces_fsi(SolventWrap *w_s, std::vector<ParticlesWrap> *w_r) {
  fsi::bind_solvent(*w_s);
  fsi::bulk(*w_r);
}

void forces() {
  SolventWrap w_s(s_pp, s_n, s_ff, cells->start, cells->count);
  std::vector<ParticlesWrap> w_r;
  if (rbcs) w_r.push_back(ParticlesWrap(r_pp, r_n, r_ff));

  clear_forces(s_ff, s_n);
  if (rbcs) clear_forces(r_ff, r_n);

  forces_dpd();
  forces_wall();

  forces_fsi(&w_s, &w_r);

  rex::bind_solutes(w_r);
  rex::pack_p();
  rex::post_p();
  rex::recv_p();

  rex::halo(); /* fsi::halo(); */

  rex::post_f();
  rex::recv_f();
}

void in_out() {
#ifdef GWRP
#include "sim.hack.h"
#endif
}

void dev2hst() { /* device to host  data transfer */
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
  Cont::rbc_dump(r_nc, &sr_pp[s_n], r_faces, r_nv, r_nt, id++);
}

void dump_grid() {
  if (!hdf5field_dumps) return;
  dev2hst();  /* TODO: do not need `r' */
  dump_field->dump(sr_pp, s_n);
}

void diag(int it) {
  int n = s_n + r_n; dev2hst();
  diagnostics(sr_pp, n, it);
}

void body_force() {
  k_sim::body_force<<<k_cnf(s_n)>>> (false, s_pp, s_ff, s_n, driving_force);

  if (!rbcs || !r_n) return;
  k_sim::body_force<<<k_cnf(r_n)>>> (true, r_pp, r_ff, r_n, driving_force);
}

#define X 0
#define Y 1
#define Z 2
#define XX 0
#define XY 1
#define XZ 2
#define YY 3
#define YZ 4
#define ZZ 5

#define YX XY
#define ZX XZ
#define ZY YZ

void update_solid() {
    CC(cudaMemcpy(r_pp_hst, r_pp, sizeof(Particle) * r_n, D2H));
    CC(cudaMemcpy(r_ff_hst, r_ff, sizeof(Force) * r_n, D2H));

    /* clear velocity */
    for (int ip = 0; ip < r_n; ++ip) {
        float *v0 = r_pp_hst[ip].v;
        v0[X] = v0[Y] = v0[Z] = 0;
    }

    solid::compute_to(r_pp_hst, r_ff_hst, r_n, r_com, /**/ r_to);
    solid::update_om(r_Iinv, r_to, /**/ r_om);
    solid::update_v(r_ff_hst, r_n, /**/ r_v);
    solid::add_v(r_pp_hst, r_n, r_v);
    solid::add_om(r_pp_hst, r_n, r_om, r_com);
    solid::rotate_e(r_e0, r_om); solid::rotate_e(r_e1, r_om); solid::rotate_e(r_e2, r_om);
    solid::gram_schmidt(r_e0, r_e1, r_e2);
    solid::update_com(r_v, /**/ r_com);
    solid::pbc_solid(r_com);
    solid::update_r(r_rr0, r_n, r_e0, r_e1, r_e2, r_com, /**/ r_pp_hst);

    CC(cudaMemcpy(r_pp, r_pp_hst, sizeof(Particle) * r_n, H2D));
}

void update_r() {
  if (r_n) update_solid();
}

void update_s() {
  k_sim::update<<<k_cnf(s_n)>>> (false, s_pp, s_ff, s_n);
}

void bounce() {
  if (!wall_created) return;
  wall::bounce(s_pp, s_n);
  if (rbcs) wall::bounce(r_pp, r_n);
}

void init() {
  CC(cudaMalloc(&r_host_av, MAX_CELLS_NUM));

  DPD::init();
  fsi::init();
  rex::init();
  if (hdf5part_dumps)
    dump_part_solvent = new H5PartDump("s.h5part");

  cells   = new CellLists(XS, YS, ZS);
  mpDeviceMalloc(&s_zip0); mpDeviceMalloc(&s_zip1);

  if (rbcs)
      mpDeviceMalloc(&r_pp); mpDeviceMalloc(&r_ff);

  wall::trunk = new Logistic::KISS;
  sdstr::init();
  mpDeviceMalloc(&s_pp); mpDeviceMalloc(&s_pp0);
  mpDeviceMalloc(&s_ff);
  mpDeviceMalloc(&r_ff); mpDeviceMalloc(&r_ff);

  s_n = ic::gen(s_pp_hst);
  CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();

  if (rbcs) {
    r_nc = Cont::setup(r_pp, r_nv, /* storage */ r_pp_hst); r_n = r_nc * r_nv;
#ifdef GWRP
    iotags_init(r_nv, r_nt, r_faces);
    iotags_domain(0, 0, 0,
		  XS, YS, ZS,
		  m::periods[0], m::periods[1], m::periods[0]);
#endif
    off::f2faces("rbc.off", r_faces);
    CC(cudaMemcpy(r_pp_hst, r_pp, sizeof(Particle) * r_n, D2H));
    solid::init(r_pp_hst, r_rr0, r_n, r_com, /**/ r_v, r_om, r_I, r_Iinv, r_e0, r_e1, r_e2);
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
  if (m::rank == 0 && !walls) printf("will take %ld steps\n", nsteps);
  if (!walls && pushtheflow) driving_force = hydrostatic_a;
  int it;
  for (it = 0; it < nsteps; ++it) {
    if (walls && it == wall_creation_stepid) {
      create_walls();
      if (rbcs && r_n) k_sim::clear_velocity<<<k_cnf(r_n)>>>(r_pp, r_n);
      if (pushtheflow) driving_force = hydrostatic_a;
    }
    distr_s();
    forces();
    dumps_diags(it);
    body_force();
    update_s();
    if (rbcs) update_r();
    bounce();
  }
}

void close() {
  delete dump_field;
  delete dump_part_solvent;
  sdstr::redist_part_close();

  delete cells;
  rex::close();
  fsi::close();
  DPD::close();

  CC(cudaFree(s_zip0));
  CC(cudaFree(s_zip1));

  CC(cudaFree(r_host_av));

  delete wall::trunk;
  CC(cudaFree(r_pp )); CC(cudaFree(r_ff ));
  CC(cudaFree(s_pp )); CC(cudaFree(s_ff ));
  CC(cudaFree(s_pp0));
}
}
