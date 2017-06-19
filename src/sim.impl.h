namespace sim {

void distr_rbc() {
    rdstr::extent(r::q.pp, r::q.nc, r::q.nv);
    dSync();
    rdstr::pack_sendcnt(r::q.pp, r::q.nc, r::q.nv);
    r::q.nc = rdstr::post(r::q.nv); r::q.n = r::q.nc * r::q.nv;
    rdstr::unpack(r::q.pp, r::q.nv);
}

void remove_rbcs_from_wall() {
  int stay[MAX_CELL_NUM];
  int nc0;
  r::q.nc = sdf::who_stays(r::q.pp, r::q.n, nc0 = r::q.nc, r::q.nv, /**/ stay);
  r::q.n = r::q.nc * r::q.nv;
  Cont::remove(r::q.pp, r::q.nv, stay, r::q.nc);
  MSG("%d/%d RBCs survived", r::q.nc, nc0);
}

void remove_solids_from_wall() {
  int stay[MAX_SOLIDS];
  int ns0;
  int nip = s::ns * s::m_dev.nv;
  s::ns = sdf::who_stays(s::i_pp_dev, nip, ns0 = s::ns, s::m_dev.nv, /**/ stay);
  s::npp = s::ns * s::nps;
  Cont::remove(s::pp,       s::nps,      stay, s::ns);
  Cont::remove(s::pp_hst,   s::nps,      stay, s::ns);

  Cont::remove(s::ss_dev,   1,           stay, s::ns);
  Cont::remove(s::ss_hst,   1,           stay, s::ns);

  Cont::remove(s::i_pp_dev, s::m_dev.nv, stay, s::ns);
  Cont::remove(s::i_pp_hst, s::m_hst.nv, stay, s::ns);
  MSG("sim.impl: %d/%d Solids survived", s::ns, ns0);
}
 

void create_walls() {
    int nold = o::n;

    dSync();
    sdf::ini();
    wall::create(&o::n, o::pp, &w::q, &w::t);
    MSG("solvent particles survived: %d/%d", o::n, nold);
    if (o::n) k_sim::clear_velocity<<<k_cnf(o::n)>>>(o::pp, o::n);
    o::cells->build(o::pp, o::n);
    sol::create_ticketZ(o::pp, o::n, &o::tz);

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
    if (rbcs) rbc::forces(r::q, /**/ r::ff);
}

void forces_dpd() {
    DPD::pack(o::pp, o::n, o::cells->start, o::cells->count);
    DPD::local_interactions(o::tz.zip0, o::tz.zip1,
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
    if (o::n)              wall::interactions(w::q, w::t, SOLVENT_TYPE, o::pp, o::n, /**/ o::ff);
    if (solids0 && s::npp) wall::interactions(w::q, w::t, SOLID_TYPE, s::pp, s::npp, /**/ s::ff);
    if (rbcs && r::q.n)    wall::interactions(w::q, w::t, SOLID_TYPE, r::q.pp, r::q.n  , /**/ r::ff);
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
    if (rbcs   ) w_r.push_back(ParticlesWrap(r::q.pp, r::q.n  , r::ff));

    clear_forces(o::ff, o::n);
    if (solids0) clear_forces(s::ff, s::npp);
    if (rbcs)    clear_forces(r::ff, r::q.n);

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
        cD2H(a::pp_hst + start, r::q.pp, r::q.n); start += r::q.n;
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
        cD2H(a::pp_hst, r::q.pp, r::q.n);
        rbc_dump(r::q.nc, a::pp_hst, r::q.tri_hst, r::q.nv, r::q.nt, id++);
    }
}

void dump_grid() {
    if (field_dumps) {
        cD2H(a::pp_hst, o::pp, o::n);
        dump_field->dump(a::pp_hst, o::n);
    }
}

void diag(int it) {
    int n = o::n + s::npp + r::q.n; dev2hst();
    diagnostics(a::pp_hst, n, it);
}

void body_force(float driving_force0) {
    k_sim::body_force<<<k_cnf(o::n)>>> (1, o::pp, o::ff, o::n, driving_force0);

    if (solids0 && s::npp)
    k_sim::body_force<<<k_cnf(s::npp)>>> (solid_mass, s::pp, s::ff, s::npp, driving_force0);

    if (rbcs && r::q.n)
    k_sim::body_force<<<k_cnf(r::q.n)>>> (rbc_mass, r::q.pp, r::ff, r::q.n, driving_force0);
}


void update_solid() {
    if (s::npp) update_solid0();
}

void update_solvent() {
    if (o::n) k_sim::update<<<k_cnf(o::n)>>> (1, o::pp, o::ff, o::n);
}

void update_rbc() {
    if (r::q.n) k_sim::update<<<k_cnf(r::q.n)>>> (rbc_mass, r::q.pp, r::ff, r::q.n);
}

void bounce() {
    if (o::n) k_sdf::bounce<<<k_cnf(o::n)>>>((float2*)o::pp, o::n);
    //if (rbcs && r::n) k_sdf::bounce<<<k_cnf(r::n)>>>((float2*)r::pp, r::n);
}

void ini() {
    if (rbcs) CC(cudaMalloc(&r::ff, MAX_PART_NUM));
    if (rbcs) rbc::alloc_quants(&r::q);
    
    rbc::setup(r::q);
    rdstr::ini();
    DPD::ini();
    fsi::ini();
    sdstr::ini();
    bbhalo::ini();
    cnt::ini();
    rex::ini();
    dump::ini();

    wall::alloc_quants(&w::q);
    wall::alloc_ticket(&w::t);

    o::cells   = new Clist(XS, YS, ZS);
    sol::alloc_ticketD(&o::td);
    sol::alloc_ticketZ(&o::tz);
    sol::alloc_work(&o::w);

    mpDeviceMalloc(&o::pp);
    mpDeviceMalloc(&o::ff);
    mpDeviceMalloc(&s::ff); mpDeviceMalloc(&s::ff);
    mpDeviceMalloc(&s::rr0);

    if (solids) {
        mrescue::ini(MAX_PART_NUM);
        s::ini();
    }

    o::n = ic::gen(o::pp_hst);
    cH2D(o::pp, o::pp_hst, o::n);
    o::cells->build(o::pp, o::n);
    create_ticketZ(o::pp, o::n, &o::tz);

    if (rbcs) {
        r::q.nc = Cont::setup(r::q.pp, r::q.nv, /* storage */ r::q.pp_hst);
        r::q.n = r::q.nc * r::q.nv;
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
    assert(r::q.n <= MAX_PART_NUM);
    sol::distr(&o::pp, &o::n, o::cells, &o::td, &o::tz, &o::w);
    if (solids0) distr_solid();
    if (rbcs)    distr_rbc();
    forces(wall0);
    dump_diag(it, wall0);
    body_force(driving_force0);
    update_solvent();
    if (solids0) update_solid();
    if (rbcs)    update_rbc();
    if (wall0) bounce();
    if (sbounce_back && solids0) bounce_solid(it);
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
    if (walls) {
        create_walls();
        wall0 = true;
        MSG("done creating walls");
    }

    MC(MPI_Barrier(m::cart));

    if (solids0) {
        cD2H(o::pp_hst, o::pp, o::n);
        s::create(o::pp_hst, &o::n);
        cH2D(o::pp, o::pp_hst, o::n);
        MC(MPI_Barrier(m::cart));
    }
    if (walls) remove_bodies();
    set_ids_solids();
    if (solids0 && s::ns) k_sim::clear_velocity<<<k_cnf(s::npp)>>>(s::pp, s::npp);
    if (rbcs    && r::q.n) k_sim::clear_velocity<<<k_cnf(r::q.n)  >>>(r::q.pp, r::q.n);
    if (pushflow) driving_force0 = driving_force;

    for (/**/; it < nsteps; ++it) step(driving_force0, wall0, it);
}

void run() {
    long nsteps = (int)(tend / dt);
    MSG0("will take %ld steps", nsteps);

    if (walls || solids) run_wall(nsteps);
    else               run_nowall(nsteps);
}

void fin() {
    sdstr::fin();

    rdstr::fin();
    bbhalo::fin();
    cnt::fin();
    DPD::fin();
    dump::fin();
    rex::fin();
    fsi::fin();

    if (solids0) mrescue::fin();

    wall::free_quants(&w::q);
    wall::free_ticket(&w::t);
    sol::free_work(&o::w);

    delete o::cells;
    delete dump_field;
    sol::free_ticketZ(&o::tz);
    sol::free_ticketD();

    CC(cudaFree(s::pp )); CC(cudaFree(s::ff )); CC(cudaFree(s::rr0));
    CC(cudaFree(o::pp )); CC(cudaFree(o::ff ));

    if (rbcs) rbc::free_quants(&r::q);
    
    if (rbcs) CC(cudaFree(r::ff));
    if (solids) s::fin();
}

}
