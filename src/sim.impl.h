namespace sim {

#define NOHOST_SOLID
    
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

void dev2hst() { /* device to host  data transfer */
  CC(cudaMemcpyAsync(sr_pp, s_pp,
		     sizeof(Particle) * s_n, D2H, 0));
  if (rbcs0)
    CC(cudaMemcpyAsync(&sr_pp[s_n], r_pp,
		       sizeof(Particle) * r_n, D2H, 0));
}

void dump_part() {
  if (!hdf5part_dumps) return;
  dev2hst();
  int sizes[2] = {s_n, r_n};
  dump_part_solvent->dump(sr_pp, sizes, 2);
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
#ifndef NOHOST_SOLID
    
    CC(cudaMemcpy(r_pp_hst, r_pp, sizeof(Particle) * r_n, D2H));
    CC(cudaMemcpy(r_ff_hst, r_ff, sizeof(Force) * r_n, D2H));
    
    solid::update_host(r_ff_hst, r_rr0_hst, r_n, nsolid, /**/ r_pp_hst, ss_hst);

    solid::reinit_f_to(nsolid, /**/ ss_hst);
    
    CC(cudaMemcpy(r_pp, r_pp_hst, sizeof(Particle) * r_n, H2D));
    
#else

    //CC(cudaMemcpy(ss_dev, ss_hst, nsolid * sizeof(Solid), H2D));
    
    solid::update_nohost(r_ff, r_rr0, r_n, nsolid, /**/ r_pp, ss_dev);

    k_solid::reinit_ft <<< k_cnf(nsolid) >>> (nsolid, /**/ ss_dev);

    CC(cudaMemcpy(ss_hst, ss_dev, sizeof(Solid), D2H));
    
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

#if 0
void bounce_solid() {
#ifndef NOHOST_SOLID
    CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));
    CC(cudaMemcpy(s_ff_hst, s_ff, sizeof(Force)    * s_n, D2H));

    solidbounce::bounce(s_ff_hst, s_n, /**/ s_pp_hst, &solid_hst);

    CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));
#else

    solidbounce::bounce_nohost(s_ff, s_n, /**/ s_pp, solid_dev);

    CC(cudaMemcpy(&solid_hst, solid_dev, sizeof(Solid), D2H));
    
#endif
}
#endif

void init_r()
{
    rex::init();
    mpDeviceMalloc(&r_pp); mpDeviceMalloc(&r_ff);

    // TMP positions of coms
    nsolid = 0;
    // float coms[2*3] = {-5.5, 0, 0,
    //                    5.5, 0, 0};

    float coms[2*3] = {0, -5.5, 0,
                       0, 5.5, 0};


    ss_hst = new Solid[nsolid];
    CC(cudaMalloc(&ss_dev, nsolid * sizeof(Solid)));

    int scount = 0;

    int *rcounts = new int[nsolid];
    for (int j = 0; j < nsolid; ++j) rcounts[j] = 0;

    CC(cudaMemcpy(s_pp_hst, s_pp, sizeof(Particle) * s_n, D2H));

    // assume same solid repeated: keep same particles
    for (int ip = 0; ip < s_n; ++ip)
    {
        Particle p = s_pp_hst[ip]; float *r0 = p.r;

        bool is_solid = false;
        for (int j = 0; j < nsolid; ++j)
        {
            float *com = coms + 3*j;

            if (solid::inside(r0[X]-com[X], r0[Y]-com[Y], r0[Z]-com[Z]))
            {
                assert(!is_solid);
                if (j == 0) r_pp_hst[rcounts[j]++] = p;
                else        ++rcounts[j];
                is_solid = true;
            }
        }
        
        if (!is_solid)
        s_pp_hst[scount++] = p;
    }
    s_n = scount;
    npsolid = rcounts[0];
    r_n = nsolid * npsolid;

    for (int j = 0; j < nsolid; ++j)
    printf("Found %d particles in solid %d\n", rcounts[j], j);
    printf("Found %d particles in solvent\n", scount);

    delete[] rcounts;

    for (int d = 0; d < 3; ++d)
    ss_hst[0].com[d] = coms[d];
    
    solid::init(r_pp_hst, npsolid, rbc_mass, /**/ r_rr0_hst, ss_hst[0].Iinv, ss_hst[0].com, ss_hst[0].e0, ss_hst[0].e1, ss_hst[0].e2, ss_hst[0].v, ss_hst[0].om);

    solid::generate(nsolid, npsolid, r_rr0_hst, coms, /**/ r_pp_hst, ss_hst);

    for (int j = 0; j < nsolid; ++j) ss_hst[j].id = j;
    
    solid::reinit_f_to(nsolid, /**/ ss_hst);
    
    CC(cudaMemcpy(ss_dev, ss_hst, nsolid * sizeof(Solid), H2D));
    CC(cudaMemcpy(r_rr0, r_rr0_hst, 3 * npsolid * sizeof(float), H2D));
  
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

  ss_hst = NULL;
  ss_dev = NULL;
  
  MC(MPI_Barrier(m::cart));
}    
    
void dumps_diags(int it) {
  if (it % steps_per_dump == 0)     dump_part();
  if (it % steps_per_dump == 0)     solid::dump(it, ss_hst, nsolid);
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
    //if (rbcs0) bounce_solid();
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

  if (ss_hst) delete[] ss_hst;
  if (ss_dev) CC(cudaFree(ss_dev));
}
#undef X
#undef Y
#undef Z
}
