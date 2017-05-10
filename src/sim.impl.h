namespace sim {

    //#define DEVICE_SOLID
    
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

static void distr_r()
{
#ifdef DEVICE_SOLID
    CC(cudaMemcpy(ss_hst, ss_dev, nsolid * sizeof(Solid), D2H));
#endif
    
    rdstr::pack_sendcnt(ss_hst, nsolid);

    nsolid = rdstr::post();
    r_n = nsolid * npsolid;

    rdstr::unpack(nsolid, /**/ ss_hst);

#ifdef DEVICE_SOLID
    CC(cudaMemcpy(ss_dev, ss_hst, nsolid * sizeof(Solid), H2D));
    solid::generate_dev(ss_dev, nsolid, r_rr0, npsolid, /**/ r_pp);
#else
    solid::generate_hst(ss_hst, nsolid, r_rr0_hst, npsolid, /**/ r_pp_hst);
    CC(cudaMemcpy(r_pp, r_pp_hst, 3 * r_n * sizeof(float), H2D));
#endif
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

void forces(bool wall_created) {
  SolventWrap w_s(s_pp, s_n, s_ff, cells->start, cells->count);
  std::vector<ParticlesWrap> w_r;
  if (rbcs0) w_r.push_back(ParticlesWrap(r_pp, r_n, r_ff));

  clear_forces(s_ff, s_n);
  if (rbcs0) clear_forces(r_ff, r_n);

  forces_dpd();
  if (wall_created) forces_wall();

  if (rbcs0) {
    forces_cnt(&w_r);
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

void dev2hst() { /* device to host  data transfer */
  CC(cudaMemcpyAsync(sr_pp, s_pp,
		     sizeof(Particle) * s_n, D2H, 0));
  if (rbcs0)
    CC(cudaMemcpyAsync(&sr_pp[s_n], r_pp,
		       sizeof(Particle) * r_n, D2H, 0));
}

void dump_part(int step)
{
    if (part_dumps)
    {
        CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));
        dump::parts(s_pp_hst, s_n, "solvent", step);

        if(rbcs0)
        {
            CC(cudaMemcpy(r_pp_hst, r_pp, sizeof(Particle) * r_n, D2H));
            dump::parts(r_pp_hst, r_n, "solid", step);
        }
    }
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
#ifndef DEVICE_SOLID
    
    CC(cudaMemcpy(r_pp_hst, r_pp, sizeof(Particle) * r_n, D2H));
    CC(cudaMemcpy(r_ff_hst, r_ff, sizeof(Force) * r_n, D2H));
    
    solid::update_hst(r_ff_hst, r_rr0_hst, r_n, nsolid, /**/ r_pp_hst, ss_hst);
    solid::update_mesh_hst(ss_hst, nsolid, m_hst, /**/ i_pp_hst);

    // for dump
    memcpy(ss_dmphst, ss_hst, nsolid * sizeof(Solid));
    
    solid::reinit_f_to(nsolid, /**/ ss_hst);
    
    CC(cudaMemcpy(r_pp, r_pp_hst, sizeof(Particle) * r_n, H2D));
    
#else    
    solid::update_dev(r_ff, r_rr0, r_n, nsolid, /**/ r_pp, ss_dev);

    // for dump
    CC(cudaMemcpy(ss_dmphst, ss_dev, nsolid * sizeof(Solid), D2H));

    k_solid::reinit_ft <<< k_cnf(nsolid) >>> (nsolid, /**/ ss_dev);
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

    mesh::bboxes(i_pp_hst, m_hst.nv, nsolid, /**/ bboxes_hst);

    // bounce on host
        
    CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));
    CC(cudaMemcpy(s_ff_hst, s_ff, sizeof(Force)    * s_n, D2H));

    mbounce::bounce_hst(s_ff_hst, s_n, nsolid, m_hst, i_pp_hst, bboxes_hst, /**/ s_pp_hst, ss_bbhst);

    CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));

    // for dump
    memcpy(ss_dmpbbhst, ss_hst, nsolid * sizeof(Solid));
}

void load_solid_mesh(const char *fname)
{
    ply::read(fname, &m_hst);

    m_dev.nv = m_hst.nv;
    m_dev.nt = m_hst.nt;
    
    CC(cudaMalloc(&(m_dev.tt), 3 * m_dev.nt * sizeof(int)));
    CC(cudaMalloc(&(m_dev.vv), 3 * m_dev.nv * sizeof(float)));

    CC(cudaMemcpy(m_dev.tt, m_hst.tt, 3 * m_dev.nt * sizeof(int), H2D));
    CC(cudaMemcpy(m_dev.vv, m_hst.vv, 3 * m_dev.nv * sizeof(float), H2D));
}

void init_solid()
{
    rex::init();
    mpDeviceMalloc(&r_pp);
    mpDeviceMalloc(&r_ff);

    bboxes_hst = new float[6 * MAX_SOLIDS];
    CC(cudaMalloc(&bboxes_dev, 6 * MAX_SOLIDS * sizeof(float)));
    
    load_solid_mesh("data/sphere.ply");
    
    ss_hst      = new Solid[MAX_SOLIDS];
    ss_bbhst    = new Solid[MAX_SOLIDS];
    ss_dmphst   = new Solid[MAX_SOLIDS];
    ss_dmpbbhst = new Solid[MAX_SOLIDS];
    
    CC(cudaMalloc(&ss_dev, MAX_SOLIDS * sizeof(Solid)));
    CC(cudaMalloc(&ss_bbdev, MAX_SOLIDS * sizeof(Solid)));
    
    CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));

    i_pp_hst = new Particle[MAX_PART_NUM];
    CC(cudaMalloc(&i_pp_dev, MAX_PART_NUM * sizeof(Particle)));

    // generate models

    ic_solid::init("ic_solid.txt", m_hst, /**/ &nsolid, &npsolid, r_rr0_hst, ss_hst, &s_n, s_pp_hst, r_pp_hst);
        
    // generate the solid particles
    
    solid::generate_hst(ss_hst, nsolid, r_rr0_hst, npsolid, /**/ r_pp_hst);
    
    solid::reinit_f_to(nsolid, /**/ ss_hst);

    r_n = nsolid * npsolid;

    solid::mesh2pp_hst(ss_hst, nsolid, m_hst, /**/ i_pp_hst);
    CC(cudaMemcpy(i_pp_dev, i_pp_hst, nsolid * m_hst.nv * sizeof(Particle), H2D));
    
    CC(cudaMemcpy(ss_dev, ss_hst, nsolid * sizeof(Solid), H2D));
    CC(cudaMemcpy(r_rr0, r_rr0_hst, 3 * npsolid * sizeof(float), H2D));
  
    CC(cudaMemcpy(r_pp, r_pp_hst, sizeof(Particle) * r_n, H2D));
    CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));

    MC(MPI_Barrier(m::cart));
}

void init() {
  DPD::init();
  fsi::init();
  rdstr::init();
  bbhalo::init();
  cnt::init();
  
  dump::init();

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

  ss_hst = NULL;
  ss_dev = NULL;

  ss_bbhst = NULL;
  ss_bbdev = NULL;

  ss_dmphst = NULL;
  ss_dmpbbhst = NULL;
  
  MC(MPI_Barrier(m::cart));
}    
    
void dumps_diags(int it) {
  if (it % steps_per_dump == 0)     dump_part(it);
  if (it % steps_per_dump == 0)     solid::dump(it, ss_dmphst, ss_dmpbbhst, nsolid); /* ss_dmpbbhst contains BB Force & Torque */
  if (it % steps_per_hdf5dump == 0) dump_grid();
  if (it % steps_per_dump == 0)     diag(it);
}

void run0(float driving_force, bool wall_created, int it) {
    distr_s();
    if (rbcs0) distr_r();
    forces(wall_created);
    dumps_diags(it);
    body_force(driving_force);
    update_s();
    if (rbcs0) update_r();
    if (wall_created) bounce();
    if (sbounce_back && rbcs0) bounce_solid();
}

void run_nowall() {
  int nsteps = (int)(tend / dt);
  if (m::rank == 0) printf("will take %ld steps\n", nsteps);
  float driving_force = pushtheflow ? hydrostatic_a : 0;
  bool wall_created = false;
  rbcs0 = false;
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
  if (rbcs0) init_solid();
  if (walls) {create_walls(); wall_created = true;}
  if (rbcs0 && r_n) k_sim::clear_velocity<<<k_cnf(r_n)>>>(r_pp, r_n);
  if (pushtheflow) driving_force = hydrostatic_a;
  
  for (/**/; it < nsteps; ++it) run0(driving_force, wall_created, it);
}

void run() {
  if (walls || rbcs) run_wall();
  else               run_nowall();
}

void close() {
  delete dump_field;
  sdstr::redist_part_close();
  rdstr::close();
  bbhalo::close();
  cnt::close();

  dump::close();
  
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

  if (m_hst.tt) delete[] m_hst.tt;
  if (m_hst.vv) delete[] m_hst.vv;
  if (m_dev.tt) CC(cudaFree(m_dev.tt));
  if (m_dev.vv) CC(cudaFree(m_dev.vv));

  if (bboxes_hst) delete[] bboxes_hst;
  if (bboxes_dev) CC(cudaFree(bboxes_dev));
  
  if (ss_hst) delete[] ss_hst;
  if (ss_dev) CC(cudaFree(ss_dev));

  if (ss_bbhst) delete[] ss_bbhst;
  if (ss_bbdev) CC(cudaFree(ss_bbdev));

  if (ss_dmphst)   delete[] ss_dmphst;
  if (ss_dmpbbhst) delete[] ss_dmpbbhst;
}
#undef X
#undef Y
#undef Z
}
