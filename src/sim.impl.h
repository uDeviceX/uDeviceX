namespace sim {

#define DEVICE_SOLID
            
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

    CC(cudaMemcpy(s::ss_hst, s::ss_dev, s::ns * sizeof(Solid), D2H));

    rdstr::pack_sendcnt <false> (s::ss_hst, s::i_pp_dev, s::ns, s::m_dev.nv);

    s::ns = rdstr::post(s::m_dev.nv);

    s::npp = s::ns * s::nps;

    rdstr::unpack <false> (s::m_dev.nv, /**/ s::ss_hst, s::i_pp_dev);

    CC(cudaMemcpy(s::ss_dev, s::ss_hst, s::ns * sizeof(Solid), H2D));

    solid::generate_dev(s::ss_dev, s::ns, s::rr0, s::nps, /**/ s::pp);

#else

    rdstr::pack_sendcnt <true> (s::ss_hst, s::i_pp_hst, s::ns, s::m_hst.nv);
    
    s::ns = rdstr::post(s::m_hst.nv);

    s::npp = s::ns * s::nps;

    rdstr::unpack <true> (s::m_hst.nv, /**/ s::ss_hst, s::i_pp_hst);

    solid::generate_hst(s::ss_hst, s::ns, s::rr0_hst, s::nps, /**/ s::pp_hst);

    CC(cudaMemcpy(s::pp, s::pp_hst, 3 * s::npp * sizeof(float), H2D));

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
  if (solids0) wall::interactions(s::pp, s::npp, s::ff);
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
  if (solids0) w_r.push_back(ParticlesWrap(s::pp, s::npp, s::ff));

  clear_forces(s_ff, s_n);
  if (solids0) clear_forces(s::ff, s::npp);

  forces_dpd();
  if (wall_created) forces_wall();

  if (solids0) {
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
  if (solids0)
    CC(cudaMemcpyAsync(&sr_pp[s_n], s::pp,
		       sizeof(Particle) * s::npp, D2H, 0));
}

void dump_part(int step)
{
    if (part_dumps)
    {
        CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));
        dump::parts(s_pp_hst, s_n, "solvent", step);

        if(solids0)
        {
            CC(cudaMemcpy(s::pp_hst, s::pp, sizeof(Particle) * s::npp, D2H));
            dump::parts(s::pp_hst, s::npp, "solid", step);
        }
    }
}

void dump_grid() {
  if (!hdf5field_dumps) return;
  dev2hst();  /* TODO: do not need `r' */
  dump_field->dump(sr_pp, s_n);
}

void diag(int it) {
  int n = s_n + s::npp; dev2hst();
  diagnostics(sr_pp, n, it);
}

void body_force(float driving_force) {
  k_sim::body_force<<<k_cnf(s_n)>>> (false, s_pp, s_ff, s_n, driving_force);

  if (!solids0 || !s::npp) return;
  k_sim::body_force<<<k_cnf(s::npp)>>> (true, s::pp, s::ff, s::npp, driving_force);
}

void update_solid() {
#ifndef DEVICE_SOLID
    
    CC(cudaMemcpy(s::pp_hst, s::pp, sizeof(Particle) * s::npp, D2H));
    CC(cudaMemcpy(s::ff_hst, s::ff, sizeof(Force) * s::npp, D2H));
    
    solid::update_hst(s::ff_hst, s::rr0_hst, s::npp, s::ns, /**/ s::pp_hst, s::ss_hst);
    solid::update_mesh_hst(s::ss_hst, s::ns, s::m_hst, /**/ s::i_pp_hst);

    // for dump
    memcpy(s::ss_dmphst, s::ss_hst, s::ns * sizeof(Solid));
    
    solid::reinit_ft_hst(s::ns, /**/ s::ss_hst);
    
    CC(cudaMemcpy(s::pp, s::pp_hst, sizeof(Particle) * s::npp, H2D));
    
#else    
    solid::update_dev(s::ff, s::rr0, s::npp, s::ns, /**/ s::pp, s::ss_dev);
    solid::update_mesh_dev(s::ss_dev, s::ns, s::m_dev, /**/ s::i_pp_dev);

    // for dump
    CC(cudaMemcpy(s::ss_dmphst, s::ss_dev, s::ns * sizeof(Solid), D2H));

    solid::reinit_ft_dev(s::ns, /**/ s::ss_dev);
#endif
}

void update_r() {
    if (s::npp) update_solid();
}

void update_s() {
  k_sim::update<<<k_cnf(s_n)>>> (false, s_pp, s_ff, s_n);
}

void bounce() {
  wall::bounce(s_pp, s_n);
  if (solids0) wall::bounce(s::pp, s::npp);
}

void bounce_solid(int it)
{
#ifndef DEVICE_SOLID

    collision::get_bboxes_hst(s::i_pp_hst, s::m_hst.nv, s::ns, /**/ s::bboxes_hst);    
    
    /* exchange solid meshes with neighbours */
    
    bbhalo::pack_sendcnt <true> (s::ss_hst, s::ns, s::i_pp_hst, s::m_hst.nv, s::bboxes_hst);
    const int nsbb = bbhalo::post(s::m_hst.nv);
    bbhalo::unpack <true> (s::m_hst.nv, /**/ s::ss_bb_hst, s::i_pp_bb_hst);
        
    build_tcells_hst(s::m_hst, s::i_pp_bb_hst, nsbb, /**/ s::tcs_hst, s::tcc_hst, s::tci_hst);
        
    CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));
    CC(cudaMemcpy(s_ff_hst, s_ff, sizeof(Force)    * s_n, D2H));

    mbounce::bounce_tcells_hst(s_ff_hst, s::m_hst, s::i_pp_bb_hst, s::tcs_hst, s::tcc_hst, s::tci_hst, s_n, /**/ s_pp_hst, s::ss_bb_hst);

    if (it % rescue_freq == 0)
    mrescue::rescue_hst(s::m_hst, s::i_pp_bb_hst, nsbb, s_n, s::tcs_hst, s::tcc_hst, s::tci_hst, /**/ s_pp_hst);
    
    CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));

    // send back fo, to
    
    bbhalo::pack_back(s::ss_bb_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::ss_hst);
    
    // for dump
    memcpy(s::ss_dmpbbhst, s::ss_hst, s::ns * sizeof(Solid));

#else // bounce on device

    collision::get_bboxes_dev(s::i_pp_dev, s::m_dev.nv, s::ns, /**/ s::bboxes_dev);
    
    CC(cudaMemcpy(s::bboxes_hst, s::bboxes_dev, 6 * s::ns * sizeof(float), D2H));
    CC(cudaMemcpy(s::ss_hst, s::ss_dev, s::ns * sizeof(Solid), D2H));

    /* exchange solid meshes with neighbours */
    
    bbhalo::pack_sendcnt <false> (s::ss_hst, s::ns, s::i_pp_dev, s::m_dev.nv, s::bboxes_hst);
    const int nsbb = bbhalo::post(s::m_dev.nv);
    bbhalo::unpack <false> (s::m_dev.nv, /**/ s::ss_bb_hst, s::i_pp_bb_dev);

    CC(cudaMemcpy(s::ss_bb_dev, s::ss_bb_hst, nsbb * sizeof(Solid), H2D));
    
    build_tcells_dev(s::m_dev, s::i_pp_bb_dev, s::ns, /**/ s::tcs_dev, s::tcc_dev, s::tci_dev);

    mbounce::bounce_tcells_dev(s_ff, s::m_dev, s::i_pp_bb_dev, s::tcs_dev, s::tcc_dev, s::tci_dev, s_n, /**/ s_pp, s::ss_dev);

    if (it % rescue_freq == 0)
    mrescue::rescue_dev(s::m_dev, s::i_pp_bb_dev, nsbb, s_n, s::tcs_dev, s::tcc_dev, s::tci_dev, /**/ s_pp);
    
    // send back fo, to

    CC(cudaMemcpy(s::ss_bb_hst, s::ss_bb_dev, nsbb * sizeof(Solid), D2H));
    
    bbhalo::pack_back(s::ss_bb_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::ss_hst);

    CC(cudaMemcpy(s::ss_dev, s::ss_hst, s::ns * sizeof(Solid), H2D));
    
    // for dump
    memcpy(s::ss_dmpbbhst, s::ss_hst, s::ns * sizeof(Solid));
    
#endif
}

void load_solid_mesh(const char *fname)
{
    ply::read(fname, &s::m_hst);

    s::m_dev.nv = s::m_hst.nv;
    s::m_dev.nt = s::m_hst.nt;
    
    CC(cudaMalloc(&(s::m_dev.tt), 3 * s::m_dev.nt * sizeof(int)));
    CC(cudaMalloc(&(s::m_dev.vv), 3 * s::m_dev.nv * sizeof(float)));

    CC(cudaMemcpy(s::m_dev.tt, s::m_hst.tt, 3 * s::m_dev.nt * sizeof(int), H2D));
    CC(cudaMemcpy(s::m_dev.vv, s::m_hst.vv, 3 * s::m_dev.nv * sizeof(float), H2D));
}

void init_solid()
{
    rex::init();
    mrescue::init(MAX_PART_NUM);
    
    mpDeviceMalloc(&s::pp);
    mpDeviceMalloc(&s::ff);
    
    load_solid_mesh("mesh_solid.ply");
    
    s::ss_hst      = new Solid[MAX_SOLIDS];
    s::ss_bb_hst   = new Solid[MAX_SOLIDS];
    s::ss_dmphst   = new Solid[MAX_SOLIDS];
    s::ss_dmpbbhst = new Solid[MAX_SOLIDS];

    s::tcs_hst = new int[XS * YS * ZS];
    s::tcc_hst = new int[XS * YS * ZS];
    s::tci_hst      = new int[27 * MAX_SOLIDS * s::m_hst.nt]; // assume 1 triangle don't overlap more than 27 cells

    CC(cudaMalloc(&s::tcs_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&s::tcc_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&s::tci_dev, 27 * MAX_SOLIDS * s::m_dev.nt * sizeof(int)));
    
    CC(cudaMalloc(&s::ss_dev, MAX_SOLIDS * sizeof(Solid)));
    CC(cudaMalloc(&s::ss_bb_dev, MAX_SOLIDS * sizeof(Solid)));
    
    CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));

    s::i_pp_hst    = new Particle[MAX_PART_NUM];
    s::i_pp_bb_hst = new Particle[MAX_PART_NUM];
    CC(cudaMalloc(   &s::i_pp_dev, MAX_PART_NUM * sizeof(Particle)));
    CC(cudaMalloc(&s::i_pp_bb_dev, MAX_PART_NUM * sizeof(Particle)));

    s::bboxes_hst = new float[6*MAX_SOLIDS];
    CC(cudaMalloc(&s::bboxes_dev, 6*MAX_SOLIDS * sizeof(float)));
    
    // generate models

    ic_solid::init("ic_solid.txt", s::m_hst, /**/ &s::ns, &s::nps, s::rr0_hst, s::ss_hst, &s_n, s_pp_hst, s::pp_hst);
        
    // generate the solid particles
    
    solid::generate_hst(s::ss_hst, s::ns, s::rr0_hst, s::nps, /**/ s::pp_hst);
    
    solid::reinit_ft_hst(s::ns, /**/ s::ss_hst);

    s::npp = s::ns * s::nps;

    solid::mesh2pp_hst(s::ss_hst, s::ns, s::m_hst, /**/ s::i_pp_hst);
    CC(cudaMemcpy(s::i_pp_dev, s::i_pp_hst, s::ns * s::m_hst.nv * sizeof(Particle), H2D));
    
    CC(cudaMemcpy(s::ss_dev, s::ss_hst, s::ns * sizeof(Solid), H2D));
    CC(cudaMemcpy(s::rr0, s::rr0_hst, 3 * s::nps * sizeof(float), H2D));
  
    CC(cudaMemcpy(s::pp, s::pp_hst, sizeof(Particle) * s::npp, H2D));
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
  mpDeviceMalloc(&s::ff); mpDeviceMalloc(&s::ff);
  mpDeviceMalloc(&s::rr0);

  s_n = ic::gen(s_pp_hst);
  CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();

  dump_field = new H5FieldDump;

  s::ss_hst = NULL;
  s::ss_dev = NULL;

  s::ss_bb_hst = NULL;
  s::ss_bb_dev = NULL;

  s::ss_dmphst = NULL;
  s::ss_dmpbbhst = NULL;

  s::tcs_hst = s::tcc_hst = s::tci_hst = NULL;
  s::tcs_dev = s::tcc_dev = s::tci_dev = NULL;
  
  MC(MPI_Barrier(m::cart));
}    
    
void dumps_diags(int it) {
  if (it % steps_per_dump == 0)     dump_part(it);
  if (it > wall_creation_stepid &&
      it % steps_per_dump == 0)     solid::dump(it, s::ss_dmphst, s::ss_dmpbbhst, s::ns, m::coords); /* s::ss_dmpbbhst contains BB Force & Torque */
  if (it % steps_per_hdf5dump == 0) dump_grid();
  if (it % steps_per_dump == 0)     diag(it);
}

void run0(float driving_force, bool wall_created, int it) {
    distr_s();
    if (solids0) distr_r();
    forces(wall_created);
    dumps_diags(it);
    body_force(driving_force);
    update_s();
    if (solids0) update_r();
    if (wall_created) bounce();
    if (sbounce_back && solids0) bounce_solid(it);
}

void run_nowall() {
  long nsteps = (int)(tend / dt);
  if (m::rank == 0) printf("will take %ld steps\n", nsteps);
  float driving_force = pushtheflow ? hydrostatic_a : 0;
  bool wall_created = false;
  solids0 = false;
  for (long it = 0; it < nsteps; ++it) run0(driving_force, wall_created, it);
}

void run_wall() {
  long nsteps = (int)(tend / dt);
  float driving_force = 0;
  bool wall_created = false;
  long it = 0;
  solids0 = false;
  for (/**/; it < wall_creation_stepid; ++it) run0(driving_force, wall_created, it);

  solids0 = solids;
  if (solids0) init_solid();
  if (walls) {create_walls(); wall_created = true;}
  if (solids0 && s::npp) k_sim::clear_velocity<<<k_cnf(s::npp)>>>(s::pp, s::npp);
  if (pushtheflow) driving_force = hydrostatic_a;
  
  for (/**/; it < nsteps; ++it) run0(driving_force, wall_created, it);
}

void run() {
  if (walls || solids) run_wall();
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
  if (solids0) {
    rex::close();
    mrescue::close();
    fsi::close();
  }
  DPD::close();

  CC(cudaFree(s_zip0));
  CC(cudaFree(s_zip1));

  delete wall::trunk;
  CC(cudaFree(s::pp )); CC(cudaFree(s::ff ));
  CC(cudaFree(s_pp )); CC(cudaFree(s_ff ));
  CC(cudaFree(s_pp0));
  CC(cudaFree(s::rr0));

  if (s::m_hst.tt) delete[] s::m_hst.tt;
  if (s::m_hst.vv) delete[] s::m_hst.vv;
  if (s::m_dev.tt) CC(cudaFree(s::m_dev.tt));
  if (s::m_dev.vv) CC(cudaFree(s::m_dev.vv));

  if (s::tcs_hst) delete[] s::tcs_hst;
  if (s::tcc_hst) delete[] s::tcc_hst;
  if (s::tci_hst) delete[] s::tci_hst;

  if (s::tcs_dev) CC(cudaFree(s::tcs_dev));
  if (s::tcc_dev) CC(cudaFree(s::tcc_dev));
  if (s::tci_dev)      CC(cudaFree(s::tci_dev));

  if (s::i_pp_hst)    delete[] s::i_pp_hst;
  if (s::i_pp_bb_hst) delete[] s::i_pp_bb_hst;
  if (s::i_pp_dev)    CC(cudaFree(s::i_pp_dev));
  if (s::i_pp_bb_dev) CC(cudaFree(s::i_pp_bb_dev));

  if (s::bboxes_hst) delete[] s::bboxes_hst;
  if (s::bboxes_dev) CC(cudaFree(s::bboxes_dev));    
  
  if (s::ss_hst) delete[] s::ss_hst;
  if (s::ss_dev) CC(cudaFree(s::ss_dev));

  if (s::ss_bb_hst) delete[] s::ss_bb_hst;
  if (s::ss_bb_dev) CC(cudaFree(s::ss_bb_dev));

  if (s::ss_dmphst)   delete[] s::ss_dmphst;
  if (s::ss_dmpbbhst) delete[] s::ss_dmpbbhst;
}
}
