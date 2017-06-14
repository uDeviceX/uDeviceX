namespace sim {

void distr_solvent() {
  x::distr(o::pp, o::pp0, o::zip0, o::zip1, &o::n, o::cells);
  std::swap(o::pp, o::pp0);
}

void distr_rbc() {
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

void remove_rbcs_from_wall() {
    int nc0 = r::nc;
    if (r::nc <= 0) return;
    DeviceBuffer<int> marks(r::n);
    k_sdf::fill_keys<<<k_cnf(r::n)>>>(r::pp, r::n, marks.D);

    std::vector<int> tmp(marks.S);
    cD2H(tmp.data(), marks.D, marks.S);
    std::vector<int> tokill;
    for (int i = 0; i < r::nc; ++i) {
	bool valid = true;
	for (int j = 0; j < r::nv && valid; ++j)
	valid &= (tmp[j + r::nv * i] == W_BULK);
	if (!valid) tokill.push_back(i);
    }

    r::nc = Cont::remove<DEV>(r::pp, r::nv, r::nc, &tokill.front(), tokill.size());
    r::n = r::nc * r::nv;
    MSG("%d/%d RBCs survived", r::nc, nc0);
}

void remove_solids_from_wall() {
    if (s::npp <= 0) return;
    int ns0 = s::ns;
    int nip = s::ns * s::m_dev.nv;
    DeviceBuffer<int> marks(nip);

    k_sdf::fill_keys<<<k_cnf(nip)>>>(s::i_pp_dev, nip, marks.D);

    std::vector<int> marks_hst(marks.S);
    cD2H(marks_hst.data(), marks.D, marks.S);
    std::vector<int> tokill;
    for (int i = 0; i < s::ns; ++i) {
	bool valid = true;
	for (int j = 0; j < s::m_dev.nv && valid; ++j)
	valid &= (marks_hst[j + s::m_dev.nv * i] == W_BULK);
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

    MSG("sim.impl: %d/%d Solids survived", s::ns, ns0);
}

void create_walls() {
    int nold = o::n;

    dSync();
    sdf::init();
    o::n = wall::init(o::pp, o::n);
    MSG("solvent particles survived: %d/%d", o::n, nold);
    if (o::n) k_sim::clear_velocity<<<k_cnf(o::n)>>>(o::pp, o::n);
    o::cells->build(o::pp, o::n);
    update_helper_arrays();
    CC( cudaPeekAtLastError() );
}

void remove_bodies() {
    if (solids) remove_solids_from_wall();
    if (rbcs)   remove_rbcs_from_wall();
}

void set_ids_solids() {
    if (solids) {
        s::ic::set_ids(s::ns, s::ss_hst);
        if (s::ns)
        cH2D(s::ss_dev, s::ss_hst, s::ns);
    }

    CC(cudaPeekAtLastError());
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
    if (o::n)              wall::interactions(SOLVENT_TYPE, o::pp, o::n, o::ff);
    if (solids0 && s::npp) wall::interactions(SOLID_TYPE, s::pp, s::npp, s::ff);
    if (rbcs && r::n)      wall::interactions(SOLID_TYPE, r::pp, r::n  , r::ff);
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

void forces(bool wall0) {
    SolventWrap w_s(o::pp, o::n, o::ff, o::cells->start, o::cells->count);
    std::vector<ParticlesWrap> w_r;
    if (solids0) w_r.push_back(ParticlesWrap(s::pp, s::npp, s::ff));
    if (rbcs   ) w_r.push_back(ParticlesWrap(r::pp, r::n  , r::ff));

    clear_forces(o::ff, o::n);
    if (solids0) clear_forces(s::ff, s::npp);
    if (rbcs)    clear_forces(r::ff, r::n);

    forces_dpd();
    if (wall0) forces_wall();
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

    dSync();
    // safety::nullify_nan(o::ff, o::n);
    // if (rbcs) safety::nullify_nan(r::ff, r::n);
    // if (solids) safety::nullify_nan(s::ff, s::npp);
}

void dev2hst() { /* device to host  data transfer */
    int start = 0;
    cD2H(a::pp_hst + start, o::pp, o::n); start += o::n;
    if (solids0) {
	cD2H(a::pp_hst + start, s::pp, s::npp); start += s::npp;
    }
    if (rbcs) {
	cD2H(a::pp_hst + start, r::pp, r::n); start += r::n;
    }
}

void dump_part(int step) {
    if (part_dumps) {
	cD2H(o::pp_hst, o::pp, o::n);
	dump::parts(o::pp_hst, o::n, "solvent", step);

	if(solids0) {
	    cD2H(s::pp_hst, s::pp, s::npp);
	    dump::parts(s::pp_hst, s::npp, "solid", step);
	}
    }
}

void dump_rbcs() {
    if (rbcs) {
	static int id = 0;
	cD2H(a::pp_hst, r::pp, r::n);
	Cont::rbc_dump(r::nc, a::pp_hst, r::faces, r::nv, r::nt, id++);
    }
}

void dump_grid() {
    if (field_dumps) {
        cD2H(a::pp_hst, o::pp, o::n);
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
    //if (rbcs && r::n) wall::bounce(r::pp, r::n);
}

void init() {
    if (rbcs) CC(cudaMalloc(&r::av, MAX_CELLS_NUM));

    rbc::setup(r::faces);
    rdstr::init();
    DPD::init();
    fsi::init();
    sdstr::init();
    x::init();
    bbhalo::init();
    cnt::init();
    rex::init();
    dump::init();

    o::cells   = new x::Clist(XS, YS, ZS);
    mpDeviceMalloc(&o::zip0); mpDeviceMalloc(&o::zip1);

    wall::trunk = new Logistic::KISS;
    mpDeviceMalloc(&o::pp); mpDeviceMalloc(&o::pp0);
    mpDeviceMalloc(&o::ff);
    mpDeviceMalloc(&s::ff); mpDeviceMalloc(&s::ff);
    mpDeviceMalloc(&s::rr0);

    if (rbcs) {
        mpDeviceMalloc(&r::pp);
        mpDeviceMalloc(&r::ff);
    }

    if (solids) {
        mrescue::init(MAX_PART_NUM);
        s::init();
    }

    o::n = ic::gen(o::pp_hst);
    cH2D(o::pp, o::pp_hst, o::n);
    o::cells->build(o::pp, o::n);
    update_helper_arrays();

    if (rbcs) {
	r::nc = Cont::setup(r::pp, r::nv, /* storage */ r::pp_hst);
	r::n = r::nc * r::nv;
    }

    dump_field = new H5FieldDump;
    MC(MPI_Barrier(m::cart));
}

void dump_diag_after(int it) { /* after wall */
  if (it % part_freq)
    solid::dump(it, s::ss_dmphst, s::ss_dmpbbhst, s::ns, m::coords);
}

void dump_diag0(int it) { /* generic dump */
  if (it % part_freq  == 0) {
    dump_part(it);
    dump_rbcs();
    diag(it);
  }
  if (it % field_freq == 0) dump_grid();
}

void dump_diag(int it, bool wall0) { /* dump and diag */
  dump_diag0(it);
  if (wall0) dump_diag_after(it);
}

void step(float driving_force0, bool wall0, int it) {
    assert(o::n <= MAX_PART_NUM);
    // safety::bound(o::pp, o::n);

    assert(r::n <= MAX_PART_NUM);
    // if (rbcs) safety::bound(r::pp, r::n);
    
    distr_solvent();
    if (solids0) distr_solid();
    if (rbcs)    distr_rbc();
    forces(wall0);
    dump_diag(it, wall0);
    body_force(driving_force0);
    update_solvent();
    if (solids0) update_solid();
    if (rbcs)    update_rbc();
    if (wall0) bounce();
    if (sbounce_back && solids0 && s::npp) bounce_solid(it);
}

void run_nowall(long nsteps) {
    float driving_force0 = pushflow ? driving_force : 0;
    bool wall0 = false;
    solids0 = false;
    for (long it = 0; it < nsteps; ++it) step(driving_force0, wall0, it);
}

void run_wall(long nsteps) {
    float driving_force0 = 0;
    bool wall0 = false;
    long it = 0;
    solids0 = false;
    for (/**/; it < wall_creation; ++it) step(driving_force0, wall0, it);

    solids0 = solids;
    if (walls) {create_walls(); wall0 = true;}
    MSG("done creating walls");
    if (solids0) {
        cD2H(o::pp_hst, o::pp, o::n);
        s::create(o::pp_hst, &o::n);
        cH2D(o::pp, o::pp_hst, o::n);
    }
    if (walls) remove_bodies();
    set_ids_solids();
    if (solids0 && s::npp) k_sim::clear_velocity<<<k_cnf(s::npp)>>>(s::pp, s::npp);
    if (rbcs && r::n)      k_sim::clear_velocity<<<k_cnf(r::n)  >>>(r::pp, r::n);
    if (pushflow) driving_force0 = driving_force;

    for (/**/; it < nsteps; ++it) step(driving_force0, wall0, it);
}

void run() {
    long nsteps = (int)(tend / dt);
    MSG0("will take %ld steps", nsteps);

    if (walls || solids) run_wall(nsteps);
    else               run_nowall(nsteps);
}

void close() {
    sdstr::close();
    x::close();
    
    rdstr::close();
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

    if (solids) s::close();
}

}
