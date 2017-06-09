namespace sim {
#define DEVICE_SOLID

void distr_solvent()
{
    odstr::pack(o::pp, o::n);
    odstr::send();
    odstr::bulk(o::n, o::cells->start, o::cells->count);
    o::n = odstr::recv_count();
    odstr::recv_unpack(o::pp0, o::zip0, o::zip1, o::n, o::cells->start, o::cells->count);
    std::swap(o::pp, o::pp0);
}

void distr_solid()
{
#ifdef DEVICE_SOLID
    CC(cudaMemcpy(s::ss_hst, s::ss_dev, s::ns * sizeof(Solid), D2H));
    sdstr::pack_sendcnt <false> (s::ss_hst, s::i_pp_dev, s::ns, s::m_dev.nv);
    s::ns = sdstr::post(s::m_dev.nv);
    s::npp = s::ns * s::nps;
    sdstr::unpack <false> (s::m_dev.nv, /**/ s::ss_hst, s::i_pp_dev);
    CC(cudaMemcpy(s::ss_dev, s::ss_hst, s::ns * sizeof(Solid), H2D));
    solid::generate_dev(s::ss_dev, s::ns, s::rr0, s::nps, /**/ s::pp);
#else
    sdstr::pack_sendcnt <true> (s::ss_hst, s::i_pp_hst, s::ns, s::m_hst.nv);
    s::ns = sdstr::post(s::m_hst.nv);
    s::npp = s::ns * s::nps;
    sdstr::unpack <true> (s::m_hst.nv, /**/ s::ss_hst, s::i_pp_hst);
    solid::generate_hst(s::ss_hst, s::ns, s::rr0_hst, s::nps, /**/ s::pp_hst);
    CC(cudaMemcpy(s::pp, s::pp_hst, 3 * s::npp * sizeof(float), H2D));
#endif
}

void distr_rbc()
{
    rdstr::extent(r::pp, r::nc, r::nv);
    dSync();
    rdstr::pack_sendcnt(r::pp, r::nc, r::nv);
    r::nc = rdstr::post(r::nv); r::n = r::nc * r::nv;
    rdstr::unpack(r::pp, r::nv);
}

void update_helper_arrays() {
    if (!o::n) return;
    CC(cudaFuncSetCacheConfig(k_sim::make_texture, cudaFuncCachePreferShared));
    k_sim::make_texture<<<(o::n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>
        (o::zip0, o::zip1, (float*)o::pp, o::n);
}

#define HST (true)
#define DEV (false)

void remove_rbcs_from_wall() {
    int nc0 = r::nc;
    if (r::nc <= 0) return;
    DeviceBuffer<int> marks(r::n);
    k_sdf::fill_keys<<<k_cnf(r::n)>>>(r::pp, r::n, marks.D);

    std::vector<int> tmp(marks.S);
    CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S, D2H));
    std::vector<int> tokill;
    for (int i = 0; i < r::nc; ++i) {
        bool valid = true;
        for (int j = 0; j < r::nv && valid; ++j)
        valid &= (tmp[j + r::nv * i] == W_BULK);
        if (!valid) tokill.push_back(i);
    }

    r::nc = Cont::remove<DEV>(r::pp, r::nv, r::nc, &tokill.front(), tokill.size());
    r::n = r::nc * r::nv;
    fprintf(stderr, "sim.impl: %04d/%04d RBCs survived\n", r::nc, nc0);
}

void remove_solids_from_wall() {
    if (s::npp <= 0) return;
    int ns0 = s::ns;
    DeviceBuffer<int> marks(s::npp);

    k_sdf::fill_keys<<<k_cnf(s::npp)>>>(s::pp, s::npp, marks.D);

    std::vector<int> marks_hst(marks.S);
    CC(cudaMemcpy(marks_hst.data(), marks.D, sizeof(int) * marks.S, D2H));
    std::vector<int> tokill;
    for (int i = 0; i < s::ns; ++i) {
        bool valid = true;
        for (int j = 0; j < s::nps && valid; ++j)
        valid &= (marks_hst[j + s::nps * i] == W_BULK);
        if (!valid) tokill.push_back(i);
    }

    int newns = 0;
    newns = Cont::remove<DEV> (s::pp,     s::nps, s::ns, &tokill.front(), tokill.size());
    newns = Cont::remove<HST> (s::pp_hst, s::nps, s::ns, &tokill.front(), tokill.size());

    newns = Cont::remove<DEV> (s::ss_dev, 1, s::ns, &tokill.front(), tokill.size());
    newns = Cont::remove<HST> (s::ss_hst, 1, s::ns, &tokill.front(), tokill.size());

    newns = Cont::remove<DEV> (s::i_pp_dev, s::m_dev.nv, s::ns, &tokill.front(), tokill.size());
    newns = Cont::remove<HST> (s::i_pp_hst, s::m_hst.nv, s::ns, &tokill.front(), tokill.size());

    s::ns = newns;
    s::npp = s::ns * s::nps;
        
    fprintf(stderr, "sim.impl: %04d/%04d Solids survived\n", s::ns, ns0);
}

#undef HST
#undef DEV

void create_walls() {
    int nold = o::n;

    dSync();
    sdf::init();
    o::n = wall::init(o::pp, o::n);
    fprintf(stderr, "%02d: solvent particles survived: %06d/06%d\n", m::rank, nold, o::n);

    if (o::n) k_sim::clear_velocity<<<k_cnf(o::n)>>>(o::pp, o::n);

    o::cells->build(o::pp, o::n, NULL, NULL);
    update_helper_arrays();

    CC( cudaPeekAtLastError() );

    if (solids) remove_solids_from_wall();
    if (rbcs)   remove_rbcs_from_wall();

    if (solids)
    {
        ic_solid::set_ids(s::ns, s::ss_hst);
        if (s::ns)
        CC(cudaMemcpy(s::ss_dev, s::ss_hst, s::ns * sizeof(Solid), H2D));
    }

    CC( cudaPeekAtLastError() );
}

void forces_rbc() {
    if (rbcs && r::n) rbc::forces(r::nc, r::pp, r::ff, r::av);
}

void forces_dpd() {
    DPD::pack(o::pp, o::n, o::cells->start, o::cells->count);
    DPD::local_interactions(o::pp, o::zip0, o::zip1,
                            o::n, o::ff, o::cells->start,
                            o::cells->count);
    DPD::post(o::pp, o::n);
    DPD::recv();
    DPD::remote_interactions(o::n, o::ff);
}

void clear_forces(Force* ff, int n) {
    if (n) CC(cudaMemsetAsync(ff, 0, sizeof(Force) * n));
}

void forces_wall() {
    if (o::n)              wall::interactions <SOLVENT_TYPE> (o::pp, o::n, o::ff);
    if (solids0 && s::npp) wall::interactions <SOLID_TYPE  > (s::pp, s::npp, s::ff);
    if (rbcs && r::n)      wall::interactions <SOLID_TYPE  > (r::pp, r::n  , r::ff);
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
    SolventWrap w_s(o::pp, o::n, o::ff, o::cells->start, o::cells->count);
    std::vector<ParticlesWrap> w_r;
    if (solids0) w_r.push_back(ParticlesWrap(s::pp, s::npp, s::ff));
    if (rbcs   ) w_r.push_back(ParticlesWrap(r::pp, r::n  , r::ff));

    clear_forces(o::ff, o::n);
    if (solids0) clear_forces(s::ff, s::npp);
    if (rbcs)    clear_forces(r::ff, r::n);

    if (o::n)         forces_dpd();
    if (wall_created) forces_wall();
    forces_rbc();

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

void dev2hst() { /* device to host  data transfer */
    int start = 0;
    CC(cudaMemcpy(a::pp_hst + start, o::pp, sizeof(Particle) * o::n, D2H)); start += o::n;
    if (solids0) {
        CC(cudaMemcpy(a::pp_hst + start, s::pp, sizeof(Particle) * s::npp, D2H)); start += s::npp;
    }
    if (rbcs) {
        CC(cudaMemcpy(a::pp_hst + start, r::pp, sizeof(Particle) * r::n, D2H)); start += r::n;
    }
}

void dump_part(int step) {
    if (part_dumps) {
        CC(cudaMemcpy(o::pp_hst, o::pp, sizeof(Particle) * o::n, D2H));
        dump::parts(o::pp_hst, o::n, "solvent", step);

        if(solids0) {
            CC(cudaMemcpy(s::pp_hst, s::pp, sizeof(Particle) * s::npp, D2H));
            dump::parts(s::pp_hst, s::npp, "solid", step);
        }
    }
}

void dump_rbcs() {
    if (rbcs) {
        static int id = 0;
        int start = o::n; if (solids0) start += s::npp;

        dev2hst();  /* TODO: do not need `s' */
        Cont::rbc_dump(r::nc, a::pp_hst + start, r::faces, r::nv, r::nt, id++);
    }
}

void dump_grid() {
    if (field_dumps) {
        dev2hst();  /* TODO: do not need `r' */
        dump_field->dump(a::pp_hst, o::n);
    }
}

void diag(int it) {
    int n = o::n + s::npp + r::n; dev2hst();
    diagnostics(a::pp_hst, n, it);
}

void body_force(float driving_force0) {
    k_sim::body_force<<<k_cnf(o::n)>>> (1, o::pp, o::ff, o::n, driving_force0);

    if (solids0 && s::npp)
    k_sim::body_force<<<k_cnf(s::npp)>>> (solid_mass, s::pp, s::ff, s::npp, driving_force0);

    if (rbcs && r::n)
    k_sim::body_force<<<k_cnf(r::n)>>> (rbc_mass, r::pp, r::ff, r::n, driving_force0);
}

void update_solid0() {
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

void update_solid() {
    if (s::npp) update_solid0();
}

void update_solvent() {
    if (o::n) k_sim::update<<<k_cnf(o::n)>>> (1, o::pp, o::ff, o::n);
}

void update_rbc() {
    if (r::n) k_sim::update<<<k_cnf(r::n)>>> (rbc_mass, r::pp, r::ff, r::n);
}

void bounce() {
    if (o::n)         wall::bounce(o::pp, o::n);
    if (rbcs && r::n) wall::bounce(r::pp, r::n);
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

    CC(cudaMemcpy(o::pp_hst, o::pp, sizeof(Particle) * o::n, D2H));
    CC(cudaMemcpy(o::ff_hst, o::ff, sizeof(Force)    * o::n, D2H));

    mbounce::bounce_tcells_hst(o::ff_hst, s::m_hst, s::i_pp_bb_hst, s::tcs_hst, s::tcc_hst, s::tci_hst, o::n, /**/ o::pp_hst, s::ss_bb_hst);

    if (it % rescue_freq == 0)
    mrescue::rescue_hst(s::m_hst, s::i_pp_bb_hst, nsbb, o::n, s::tcs_hst, s::tcc_hst, s::tci_hst, /**/ o::pp_hst);

    CC(cudaMemcpy(o::pp, o::pp_hst, sizeof(Particle) * o::n, H2D));

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

    mbounce::bounce_tcells_dev(o::ff, s::m_dev, s::i_pp_bb_dev, s::tcs_dev, s::tcc_dev, s::tci_dev, o::n, /**/ o::pp, s::ss_dev);

    if (it % rescue_freq == 0)
    mrescue::rescue_dev(s::m_dev, s::i_pp_bb_dev, nsbb, o::n, s::tcs_dev, s::tcc_dev, s::tci_dev, /**/ o::pp);

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
    s::tci_hst = new int[27 * MAX_SOLIDS * s::m_hst.nt]; // assume 1 triangle don't overlap more than 27 cells

    CC(cudaMalloc(&s::tcs_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&s::tcc_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&s::tci_dev, 27 * MAX_SOLIDS * s::m_dev.nt * sizeof(int)));

    CC(cudaMalloc(&s::ss_dev,    MAX_SOLIDS * sizeof(Solid)));
    CC(cudaMalloc(&s::ss_bb_dev, MAX_SOLIDS * sizeof(Solid)));

    CC(cudaMemcpy(o::pp_hst, o::pp, sizeof(Particle) * o::n, D2H));

    s::i_pp_hst    = new Particle[MAX_PART_NUM];
    s::i_pp_bb_hst = new Particle[MAX_PART_NUM];
    CC(cudaMalloc(   &s::i_pp_dev, MAX_PART_NUM * sizeof(Particle)));
    CC(cudaMalloc(&s::i_pp_bb_dev, MAX_PART_NUM * sizeof(Particle)));

    s::bboxes_hst = new float[6*MAX_SOLIDS];
    CC(cudaMalloc(&s::bboxes_dev, 6*MAX_SOLIDS * sizeof(float)));

    // generate models

    ic_solid::init("ic_solid.txt", s::m_hst, /**/ &s::ns, &s::nps, s::rr0_hst, s::ss_hst, &o::n, o::pp_hst, s::pp_hst);

    // generate the solid particles

    solid::generate_hst(s::ss_hst, s::ns, s::rr0_hst, s::nps, /**/ s::pp_hst);
    solid::reinit_ft_hst(s::ns, /**/ s::ss_hst);
    s::npp = s::ns * s::nps;

    solid::mesh2pp_hst(s::ss_hst, s::ns, s::m_hst, /**/ s::i_pp_hst);
    CC(cudaMemcpy(s::i_pp_dev, s::i_pp_hst, s::ns * s::m_hst.nv * sizeof(Particle), H2D));

    CC(cudaMemcpy(s::ss_dev, s::ss_hst, s::ns * sizeof(Solid), H2D));
    CC(cudaMemcpy(s::rr0, s::rr0_hst, 3 * s::nps * sizeof(float), H2D));

    CC(cudaMemcpy(s::pp, s::pp_hst, sizeof(Particle) * s::npp, H2D));
    CC(cudaMemcpy(o::pp, o::pp_hst, sizeof(Particle) * o::n, H2D));

    MC(MPI_Barrier(m::cart));
}

void init() {
    CC(cudaMalloc(&r::av, MAX_CELLS_NUM));

    rbc::setup(r::faces);
    rdstr::ini();
    DPD::init();
    fsi::init();
    sdstr::init();
    bbhalo::init();
    cnt::init();
    rex::init();

    dump::init();

    o::cells   = new CellLists(XS, YS, ZS);
    mpDeviceMalloc(&o::zip0); mpDeviceMalloc(&o::zip1);

    wall::trunk = new Logistic::KISS;
    odstr::init();
    mpDeviceMalloc(&o::pp); mpDeviceMalloc(&o::pp0);
    mpDeviceMalloc(&o::ff);
    mpDeviceMalloc(&s::ff); mpDeviceMalloc(&s::ff);
    mpDeviceMalloc(&s::rr0);

    if (rbcs)
    {
        mpDeviceMalloc(&r::pp);
        mpDeviceMalloc(&r::ff);
    }

    o::n = ic::gen(o::pp_hst);
    CC(cudaMemcpy(o::pp, o::pp_hst, sizeof(Particle) * o::n, H2D));
    o::cells->build(o::pp, o::n, NULL, NULL);
    update_helper_arrays();

    if (rbcs) {
        r::nc = Cont::setup(r::pp, r::nv, /* storage */ r::pp_hst);
        r::n = r::nc * r::nv;
    }

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
    if (it > wall_creation && it % part_freq == 0) {
        /* s::ss_dmpbbhst contains BB Force & Torque */
        solid::dump(it, s::ss_dmphst, s::ss_dmpbbhst, s::ns, m::coords);
        dump_rbcs();
    }

    if (it % part_freq == 0)    dump_part(it);
    if (it % field_freq == 0)   dump_grid();
    if (it % part_freq == 0)    diag(it);
}

void run0(float driving_force0, bool wall_created, int it) {
    distr_solvent();
    if (solids0) distr_solid();
    if (rbcs)    distr_rbc();
    forces(wall_created);
    dumps_diags(it);
    body_force(driving_force0);
    update_solvent();
    if (solids0) update_solid();
    if (rbcs)    update_rbc();
    if (wall_created) bounce();
    if (sbounce_back && solids0 && s::npp) bounce_solid(it);
}

void run_nowall(long nsteps) {
    float driving_force0 = pushflow ? driving_force : 0;
    bool wall_created = false;
    solids0 = false;
    for (long it = 0; it < nsteps; ++it) run0(driving_force0, wall_created, it);
}

void run_wall(long nsteps) {
    float driving_force0 = 0;
    bool wall_created = false;
    long it = 0;
    solids0 = false;
    for (/**/; it < wall_creation; ++it) run0(driving_force0, wall_created, it);

    solids0 = solids;
    if (solids0) init_solid();
    if (walls) {create_walls(); wall_created = true;}
    printf("%d: done creating walls\n", m::rank);
    if (solids0 && s::npp) k_sim::clear_velocity<<<k_cnf(s::npp)>>>(s::pp, s::npp);
    if (rbcs && r::n)      k_sim::clear_velocity<<<k_cnf(r::n)  >>>(r::pp, r::n);
    if (pushflow) driving_force0 = driving_force;

    for (/**/; it < nsteps; ++it) run0(driving_force0, wall_created, it);
}

void run() {
    long nsteps = (int)(tend / dt);
    if (m::rank == 0) printf("will take %ld steps\n", nsteps);

    if (walls || solids) run_wall(nsteps);
    else               run_nowall(nsteps);
}

void close() {

    odstr::redist_part_close();
    sdstr::close();
    rdstr::fin();
    bbhalo::close();
    cnt::close();
    DPD::close();
    dump::close();
    rex::close();
    fsi::close();

    if (solids0)
    mrescue::close();

    delete o::cells;
    delete dump_field;
    delete wall::trunk;

    CC(cudaFree(o::zip0));
    CC(cudaFree(o::zip1));

    CC(cudaFree(s::pp )); CC(cudaFree(s::ff )); CC(cudaFree(s::rr0));
    CC(cudaFree(o::pp )); CC(cudaFree(o::ff )); CC(cudaFree(o::pp0));

    if (rbcs)
    {
        CC(cudaFree(r::pp));
        CC(cudaFree(r::ff));
        CC(cudaFree(r::av));
    }

    if (solids)
    {
        delete[] s::m_hst.tt;      CC(cudaFree(s::m_dev.tt));
        delete[] s::m_hst.vv;      CC(cudaFree(s::m_dev.vv));

        delete[] s::tcs_hst;       CC(cudaFree(s::tcs_dev));
        delete[] s::tcc_hst;       CC(cudaFree(s::tcc_dev));
        delete[] s::tci_hst;       CC(cudaFree(s::tci_dev));

        delete[] s::i_pp_hst;      CC(cudaFree(s::i_pp_dev));
        delete[] s::i_pp_bb_hst;   CC(cudaFree(s::i_pp_bb_dev));

        delete[] s::bboxes_hst;    CC(cudaFree(s::bboxes_dev));
        delete[] s::ss_hst;        CC(cudaFree(s::ss_dev));
        delete[] s::ss_bb_hst;     CC(cudaFree(s::ss_bb_dev));
        delete[] s::ss_dmphst;     delete[] s::ss_dmpbbhst;
    }
}
}
