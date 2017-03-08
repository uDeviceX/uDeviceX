namespace sim {

static void distr_s() {
  sdstr::pack(s_pp, s_n);
  sdstr::send();
  sdstr::bulk(s_n, cells->start, cells->count);
  s_n = sdstr::recv_count();
  sdstr::recv_unpack(s_pp0, s_zip0, s_zip1, s_n, cells->start, cells->count);
  std::swap(s_pp, s_pp0);
}

static void distr_r() {
  rdstr::extent(r_pp, r_nc, r_nv);
  rdstr::pack_sendcnt(r_pp, r_nc, r_nv);
  r_nc = rdstr::post(r_nv); r_n = r_nc * r_nv;
  rdstr::unpack(r_pp, r_nc, r_nv);
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

void init_I() {
    CC(cudaMemcpy(r_pp_hst, r_pp, sizeof(Particle) * r_n, D2H));

    int ip, c;
    float *r0, x, y, z;

    float com[3] = {0, 0, 0};
    for (ip = 0; ip < r_n; ++ip) {
        r0 = r_pp_hst[ip].r;
        com[X] += r0[X]; com[Y] += r0[Y]; com[Z] += r0[Z];
    }
    com[X] /= r_n; com[Y] /= r_n; com[Z] /= r_n;

    float *I = r_I;
    for (c = 0; c < 6; ++c) I[c] = 0;
    for (ip = 0; ip < r_n; ++ip) {
        r0 = r_pp_hst[ip].r;
        x = r0[X]-com[X]; y = r0[Y]-com[Y]; z = r0[Z]-com[Z];
        I[XX] += y*y + z*z;
        I[YY] += z*z + x*x;
        I[ZZ] += x*x + y*y;
        I[XY] -= x*y;
        I[XZ] -= z*x;
        I[YZ] -= y*z;
    }
    for (c = 0; c < 6; ++c) I[c] /= r_m;
}

void init_solid() {
    r_m = rbc_mass*r_n;
    r_v[0] = 0; r_v[1] = 0; r_v[2] = 0; 
    init_I();
}

void update_solid() {
    CC(cudaMemcpy(r_pp_hst, r_pp, sizeof(Particle) * r_n, D2H));
    CC(cudaMemcpy(r_ff_hst, r_ff, sizeof(Force) * r_n, D2H));

    int ip;
    float *r0, *v0, *f0, x, y, z, fx, fy, fz;

    /* update force */
    r_f[X] = X; r_f[Y] = X; r_f[Z] = X; 
    for (ip = X; ip < r_n; ++ip) {
        f0 = r_ff_hst[ip].f;
        r_f[X] += f0[X]; r_f[Y] += f0[Y]; r_f[Z] += f0[Z];
    }

    /* compute COM */
    float com[3] = {0, 0, 0};
    for (ip = 0; ip < r_n; ++ip) {
        r0 = r_pp_hst[ip].r;
        com[X] += r0[X]; com[Y] += r0[Y]; com[Z] += r0[Z];
    }
    com[X] /= r_n; com[Y] /= r_n; com[Z] /= r_n;
    printf("COM: %g %g %g\n", com[X], com[Y], com[Z]);

    /* update torque */
    r_to[X] = r_to[Y] = r_to[Z] = 0;
    for (ip = 0; ip < r_n; ++ip) {
        r0 = r_pp_hst[ip].r; f0 = r_ff_hst[ip].f;
        x = r0[X]-com[X]; y = r0[Y]-com[Y]; z = r0[Z]-com[Z];
        fx = f0[X]; fy = f0[Y]; fz = f0[Z];
        r_to[X] += y*fz - z*fy;
        r_to[Y] += z*fx - x*fz;
        r_to[Z] += x*fy - y*fx;
    }

    /* update linear velocity */
    float sc = 1./r_m*dt;
    r_v[X] += r_f[X]*sc; r_v[Y] += r_f[Y]*sc; r_v[Z] += r_f[Z]*sc;

    for (ip = 0; ip < r_n; ++ip) {
        v0 = r_pp_hst[ip].v;
        v0[X] = r_v[X]; v0[Y] = r_v[Y]; v0[Z] = r_v[Z];
    }

    for (ip = 0; ip < r_n; ++ip) {
        r0 = r_pp_hst[ip].r;
        v0 = r_pp_hst[ip].v;
        r0[X] += v0[X]*dt; r0[Y] += v0[Y]*dt; r0[Z] += v0[Z]*dt;
    }

    CC(cudaMemcpy(r_pp, r_pp_hst, sizeof(Particle) * r_n, H2D));
    //CC(cudaMemcpy(r_ff, r_ff_hst, sizeof(Force) * r_n, H2D));
    
    //k_sim::update<<<k_cnf(r_n)>>> (true,  r_pp, r_ff, r_n);
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
  invert_matrix();
  exit(0);

  CC(cudaMalloc(&r_host_av, MAX_CELLS_NUM));

  off::f2faces("rbc.off", r_faces);
  rdstr::init();
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
    init_solid();
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
    if (rbcs) distr_r();
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
  rdstr::close();

  CC(cudaFree(s_zip0));
  CC(cudaFree(s_zip1));

  CC(cudaFree(r_host_av));

  delete wall::trunk;
  CC(cudaFree(r_pp )); CC(cudaFree(r_ff ));
  CC(cudaFree(s_pp )); CC(cudaFree(s_ff ));
  CC(cudaFree(s_pp0));
}
}
