namespace sim {

#define X 0
#define Y 1
#define Z 2
    
static void distr_s() {
  sdstr::pack(s_pp, s_n);
  sdstr::send();
  sdstr::bulk(s_n, cells->start, cells->count);
  s_n = sdstr::recv_count();
  sdstr::recv_unpack(s_pp0, s_zip0, s_zip1, s_n, cells->start, cells->count);
  std::swap(s_pp, s_pp0);
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

  k_sim::clear_velocity<<<k_cnf(s_n)>>>(s_pp, s_n);
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();
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
  if (rbcs0) wall::interactions(r_pp, r_n, r_ff);
  wall::interactions(s_pp, s_n, s_ff);
}

void forces_fsi(SolventWrap *w_s, std::vector<ParticlesWrap> *w_r) {
  fsi::bind_solvent(*w_s);
  fsi::bulk(*w_r);
}

void forces(bool wall_created) {
  SolventWrap w_s(s_pp, s_n, s_ff, cells->start, cells->count);
  std::vector<ParticlesWrap> w_r;
  if (rbcs0) w_r.push_back(ParticlesWrap(r_pp, r_n, r_ff));

  clear_forces(s_ff, s_n);
  if (rbcs0) clear_forces(r_ff, r_n);

  forces_dpd();
  if (wall_created) forces_wall();

  if (rbcs0) {
    forces_fsi(&w_s, &w_r);

    rex::bind_solutes(w_r);
    rex::pack_p();
    rex::post_p();
    rex::recv_p();

    rex::halo(); /* fsi::halo(); */

    rex::post_f();
    rex::recv_f();
  }
}

void in_out() {
#ifdef GWRP
#include "sim.hack.h"
#endif
}

void dev2hst() { /* device to host  data transfer */
  CC(cudaMemcpyAsync(sr_pp, s_pp,
		     sizeof(Particle) * s_n, D2H, 0));
  if (rbcs0)
    CC(cudaMemcpyAsync(&sr_pp[s_n], r_pp,
		       sizeof(Particle) * r_n, D2H, 0));
}

void dump_part() {
  if (!hdf5part_dumps) return;
  dev2hst(); /* TODO: do not need `r' */
  int n = s_n + r_n;
  dump_part_solvent->dump(sr_pp, n);
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

void body_force(float driving_force) {
  k_sim::body_force<<<k_cnf(s_n)>>> (false, s_pp, s_ff, s_n, driving_force);

  if (!rbcs0 || !r_n) return;
  k_sim::body_force<<<k_cnf(r_n)>>> (true, r_pp, r_ff, r_n, driving_force);
}

void update_solid() {
#if 0 // Host only
    
    CC(cudaMemcpy(r_pp_hst, r_pp, sizeof(Particle) * r_n, D2H));
    CC(cudaMemcpy(r_ff_hst, r_ff, sizeof(Force) * r_n, D2H));
    
    solid::update(r_ff_hst, r_rr0_hst, r_n, /**/ r_pp_hst, &solid_hst);

    solid::reinit_f_to(/**/ solid_hst.fo, solid_hst.to);
    
    CC(cudaMemcpy(r_pp, r_pp_hst, sizeof(Particle) * r_n, H2D));
    
#else

    CC(cudaMemcpy(solid_dev, &solid_hst, sizeof(Solid), H2D));
    
    solid::update_nohost(r_ff, r_rr0, r_n, /**/ r_pp, solid_dev);

    k_solid::reinit_ft <<<1, 1>>> (solid_dev->fo, solid_dev->to);

    CC(cudaMemcpy(&solid_hst, solid_dev, sizeof(Solid), D2H));
    
#endif
}

void update_r() {
  if (r_n) update_solid();
}

void update_s() {
  k_sim::update<<<k_cnf(s_n)>>> (false, s_pp, s_ff, s_n);
}

void bounce() {
  wall::bounce(s_pp, s_n);
  if (rbcs0) wall::bounce(r_pp, r_n);
}

void bounce_solid() {
#if 0 // Host only
    CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));
    CC(cudaMemcpy(s_ff_hst, s_ff, sizeof(Force)    * s_n, D2H));

    solidbounce::bounce(s_ff_hst, s_n, /**/ s_pp_hst, &solid_hst);

    CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));
#else

    solidbounce::bounce_nohost(s_ff, s_n, /**/ s_pp, solid_dev);

    CC(cudaMemcpy(&solid_hst, solid_dev, sizeof(Solid), D2H));
    
#endif
}

void init_r() {
  rex::init();
  mpDeviceMalloc(&r_pp); mpDeviceMalloc(&r_ff);
  int ip, is, ir;

  CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));
  
  for (ip = is = ir = 0; ip < s_n; ++ip) {
    Particle p = s_pp_hst[ip]; float *r0 = p.r;
    if (solid::inside(r0[X], r0[Y], r0[Z])) r_pp_hst[ir++] = p;
    else                                    s_pp_hst[is++] = p;
  }
  r_n = ir; s_n = is;

  for (ip = 0; ip < r_n; ++ip) {
    float *v0 = r_pp_hst[ip].v;
    lastbit::set(v0[0], true);
  }

  solid_hst.mass = rbc_mass;
  
  solid::init(r_pp_hst, r_n, solid_hst.mass, /**/ r_rr0_hst, solid_hst.Iinv, solid_hst.com, solid_hst.e0, solid_hst.e1, solid_hst.e2, solid_hst.v, solid_hst.om);

  CC(cudaMemcpy(solid_dev, &solid_hst, sizeof(Solid), H2D));
  CC(cudaMemcpy(r_rr0, r_rr0_hst, 3 * r_n * sizeof(float), H2D));

  solid::reinit_f_to(/**/ solid_hst.fo, solid_hst.to);
  
  CC(cudaMemcpy(r_pp, r_pp_hst, sizeof(Particle) * r_n, H2D));
  CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));

  MC(MPI_Barrier(m::cart));
}

void init() {
  DPD::init();
  fsi::init();
  if (hdf5part_dumps)
    dump_part_solvent = new H5PartDump("s.h5part");

  cells   = new CellLists(XS, YS, ZS);
  mpDeviceMalloc(&s_zip0); mpDeviceMalloc(&s_zip1);

  wall::trunk = new Logistic::KISS;
  sdstr::init();
  mpDeviceMalloc(&s_pp); mpDeviceMalloc(&s_pp0);
  mpDeviceMalloc(&s_ff);
  mpDeviceMalloc(&r_ff); mpDeviceMalloc(&r_ff);
  mpDeviceMalloc(&r_rr0);

  s_n = ic::gen(s_pp_hst);
  CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();

  dump_field = new H5FieldDump;

  CC(cudaMalloc(&solid_dev, sizeof(Solid)));
  
  MC(MPI_Barrier(m::cart));
}

void dumps_diags(int it) {
  if (it % steps_per_dump == 0)     in_out();
  if (it % steps_per_dump == 0)     dump_part();
  if (it % steps_per_dump == 0)     solid::dump(it, solid_hst.com, solid_hst.v, solid_hst.om, solid_hst.to);
  if (it % steps_per_hdf5dump == 0) dump_grid();
  if (it % steps_per_dump == 0)     diag(it);
}

void run0(float driving_force, bool wall_created, int it) {
    distr_s();
    forces(wall_created);
    dumps_diags(it);
    body_force(driving_force);
    update_s();
    if (rbcs0) update_r();
    if (wall_created) bounce();
    if (rbcs0) bounce_solid();
}

void run_nowall() {
  int nsteps = (int)(tend / dt);
  if (m::rank == 0) printf("will take %ld steps\n", nsteps);
  float driving_force = pushtheflow ? hydrostatic_a : 0;
  bool wall_created = false;
  rbcs0 = rbcs;
  if (rbcs0) init_r();
  for (int it = 0; it < nsteps; ++it) run0(driving_force, wall_created, it);
}

void run_wall() {
  int nsteps = (int)(tend / dt);
  float driving_force = 0;
  bool wall_created = false;
  int it = 0;
  rbcs0 = false;
  for (/**/; it < wall_creation_stepid; ++it) run0(driving_force, wall_created, it);

  rbcs0 = rbcs;
  if (rbcs0) init_r();
  create_walls(); wall_created = true;
  if (rbcs0 && r_n) k_sim::clear_velocity<<<k_cnf(r_n)>>>(r_pp, r_n);
  if (pushtheflow) driving_force = hydrostatic_a;

  for (/**/; it < nsteps; ++it) run0(driving_force, wall_created, it);
}

void run() {
  if (walls) run_wall();
  else       run_nowall();
}

void close() {
  delete dump_field;
  delete dump_part_solvent;
  sdstr::redist_part_close();

  delete cells;
  if (rbcs0) {
    rex::close();
    fsi::close();
  }
  DPD::close();

  CC(cudaFree(s_zip0));
  CC(cudaFree(s_zip1));

  delete wall::trunk;
  CC(cudaFree(r_pp )); CC(cudaFree(r_ff ));
  CC(cudaFree(s_pp )); CC(cudaFree(s_ff ));
  CC(cudaFree(s_pp0));
  CC(cudaFree(r_rr0));

  CC(cudaFree(solid_dev));
}
#undef X
#undef Y
#undef Z
}
