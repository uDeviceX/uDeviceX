namespace sim {
/* see bund.cu for more sim:: functions */

void create_walls() {
    int nold = o::n;

    dSync();
    sdf::ini();
    wall::gen_quants(&o::n, o::pp, &w::q);
    wall::gen_ticket(w::q, &w::t);
    MSG("solvent particles survived: %d/%d", o::n, nold);
    if (o::n) k_sim::clear_velocity<<<k_cnf(o::n)>>>(o::pp, o::n);
    o::cells->build(o::pp, o::n);
    flu::create_ticketZ(o::pp, o::n, &o::tz);

    CC( cudaPeekAtLastError() );
}

void set_ids_solids() {
    rig::set_ids(s::q);
    CC(cudaPeekAtLastError());
}

void update_solid() {
    if (s::q.n) update_solid0();
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
    if (rbcs) {
        CC(cudaMalloc(&r::ff, MAX_PART_NUM));
        rbc::alloc_quants(&r::q);
        rbc::setup("rbc.off", &r::q);
        rbc::setup_textures(r::q, &r::tt);
    }
        
    rdstr::ini();
    dpd::ini();
    fsi::ini();
    sdstr::ini();
    bbhalo::ini();
    cnt::ini();
    rex::ini();
    dump::ini();

    wall::alloc_quants(&w::q);
    wall::alloc_ticket(&w::t);

    o::cells   = new Clist(XS, YS, ZS);
    flu::alloc_ticketD(&o::td);
    flu::alloc_ticketZ(&o::tz);
    flu::alloc_work(&o::w);

    mpDeviceMalloc(&o::pp);
    mpDeviceMalloc(&o::ff);

    if (solids) {
        mrescue::ini(MAX_PART_NUM);
        rig::alloc_quants(&s::q);
        rig::alloc_ticket(&s::t);
        s::ff_hst = new Force[MAX_PART_NUM];
        CC(cudaMalloc(&s::ff, MAX_PART_NUM * sizeof(Force)));
    }

    o::n = ic::gen(o::pp_hst);
    cH2D(o::pp, o::pp_hst, o::n);
    o::cells->build(o::pp, o::n);
    create_ticketZ(o::pp, o::n, &o::tz);

    if (rbcs) rbc::setup_from_pos("rbc.off", "rbcs-ic.txt", /**/ &r::q);
    
    dump_field = new H5FieldDump;
    MC(MPI_Barrier(m::cart));
}

void step(float driving_force0, bool wall0, int it) {
    assert(o::n <= MAX_PART_NUM);
    assert(r::q.n <= MAX_PART_NUM);
    flu::distr(&o::pp, &o::n, o::cells, &o::td, &o::tz, &o::w);
    if (solids0) distr_solid();
    if (rbcs)    distr_rbc();
    forces(wall0);
    dump_diag0(it);
    if (wall0) dump_diag_after(it);
    body_force(driving_force0);
    update_solvent();
    if (solids0) update_solid();
    if (rbcs)    update_rbc();
    if (wall0) bounce();
    if (sbounce_back && solids0) bounce_solid(it);
}

void run_simple(long nsteps) {
    float driving_force0 = pushflow ? driving_force : 0;
    bool wall0 = false;
    solids0 = false;
    for (long it = 0; it < nsteps; ++it) step(driving_force0, wall0, it);
}

void run_complex(long nsteps) {
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

        rig::create(/*io*/ o::pp_hst, &o::n, /**/ &s::q);

        MC(l::m::Barrier(m::cart));
        
        rig::gen_pp_hst(s::q);
        rig::gen_ipp_hst(s::q);
        rig::cpy_H2D(s::q);

        cH2D(o::pp, o::pp_hst, o::n);
        MC(l::m::Barrier(m::cart));
        MSG("created %d solids.", s::q.ns);
    }
    if (walls) remove_bodies();
    if (solids) set_ids_solids();
    if (solids0 && s::q.n) k_sim::clear_velocity<<<k_cnf(s::q.n)>>>(s::q.pp, s::q.n);
    if (rbcs    && r::q.n) k_sim::clear_velocity<<<k_cnf(r::q.n)>>>(r::q.pp, r::q.n);
    if (pushflow) driving_force0 = driving_force;

    for (/**/; it < nsteps; ++it) step(driving_force0, wall0, it);
}

void run() {
    long nsteps = (int)(tend / dt);
    MSG0("will take %ld steps", nsteps);

    if (walls || solids) run_complex(nsteps);
    else                 run_simple(nsteps);
}

void fin() {
    sdstr::fin();
    rdstr::fin();
    bbhalo::fin();
    cnt::fin();
    dpd::fin();
    dump::fin();
    rex::fin();
    fsi::fin();

    if (solids) mrescue::fin();

    wall::free_quants(&w::q);
    wall::free_ticket(&w::t);
    flu::free_work(&o::w);

    delete o::cells;
    delete dump_field;
    flu::free_ticketZ(&o::tz);
    flu::free_ticketD(&o::td);

    if (solids) {
        rig::free_quants(&s::q);
        rig::free_ticket(&s::t);
        CC(cudaFree(s::ff)); delete[] s::ff_hst;
    }

    if (rbcs) {
        rbc::free_quants(&r::q);
        rbc::destroy_textures(&r::tt);
        CC(cudaFree(r::ff));
    }
}

}
