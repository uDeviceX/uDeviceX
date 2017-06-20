namespace sim {
/* see bund.cu for more sim:: functions */
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
  int nip = s::q.ns * s::q.m_dev.nv;
  s::q.ns = sdf::who_stays(s::q.i_pp, nip, ns0 = s::q.ns, s::q.m_dev.nv, /**/ stay);
  s::q.n  = s::q.ns * s::q.nps;
  Cont::remove(s::q.pp,       s::q.nps,      stay, s::q.ns);
  Cont::remove(s::q.pp_hst,   s::q.nps,      stay, s::q.ns);

  Cont::remove(s::q.ss,       1,           stay, s::q.ns);
  Cont::remove(s::q.ss_hst,   1,           stay, s::q.ns);

  Cont::remove(s::q.i_pp,     s::q.m_dev.nv, stay, s::q.ns);
  Cont::remove(s::q.i_pp_hst, s::q.m_hst.nv, stay, s::q.ns);
  MSG("sim.impl: %d/%d Solids survived", s::q.ns, ns0);
}
 

void create_walls() {
    int nold = o::n;

    dSync();
    sdf::ini();
    wall::create(&o::n, o::pp, &w::q, &w::t);
    MSG("solvent particles survived: %d/%d", o::n, nold);
    if (o::n) k_sim::clear_velocity<<<k_cnf(o::n)>>>(o::pp, o::n);
    o::cells->build(o::pp, o::n);
    flu::create_ticketZ(o::pp, o::n, &o::tz);

    CC( cudaPeekAtLastError() );
}

void remove_bodies() {
    if (solids) remove_solids_from_wall();
    if (rbcs)   remove_rbcs_from_wall();
}

void set_ids_solids() {
    if (solids) rig::set_ids(s::q);
    CC(cudaPeekAtLastError());
}

void forces_rbc() {
    if (rbcs) rbc::forces(r::q, r::tt, /**/ r::ff);
}

void forces_dpd() {
    dpd::pack(o::pp, o::n, o::cells->start, o::cells->count);
    /* :TODO: breaks a contract with hiwi */
    dpd::local_interactions(o::tz.zip0, o::tz.zip1,
                            o::n, o::cells->start, o::cells->count,
                            /**/ o::ff);
    dpd::post(o::pp, o::n);
    dpd::recv();
    dpd::remote_interactions(o::n, o::ff);
}

void clear_forces(Force* ff, int n) {
    if (n) CC(cudaMemsetAsync(ff, 0, sizeof(Force) * n));
}

void forces_wall() {
    if (o::n)              wall::interactions(w::q, w::t, SOLVENT_TYPE, o::pp, o::n,   /**/ o::ff);
    if (solids0 && s::q.n) wall::interactions(w::q, w::t, SOLID_TYPE, s::q.pp, s::q.n, /**/ s::ff);
    if (rbcs && r::q.n)    wall::interactions(w::q, w::t, SOLID_TYPE, r::q.pp, r::q.n, /**/ r::ff);
}

void forces_cnt(std::vector<ParticlesWrap> *w_r) {
    cnt::build_cells(*w_r);
    cnt::bulk(*w_r);
}

void forces_fsi(SolventWrap *w_s, std::vector<ParticlesWrap> *w_r) {
    fsi::bind_solvent(*w_s);
    fsi::bulk(*w_r);
}

void forces(bool wall0) {
    SolventWrap w_s(o::pp, o::n, o::ff, o::cells->start, o::cells->count);
    std::vector<ParticlesWrap> w_r;
    if (solids0) w_r.push_back(ParticlesWrap(s::q.pp, s::q.n, s::ff));
    if (rbcs   ) w_r.push_back(ParticlesWrap(r::q.pp, r::q.n, r::ff));

    clear_forces(o::ff, o::n);
    if (solids0) clear_forces(s::ff, s::q.n);
    if (rbcs)    clear_forces(r::ff, r::q.n);

    forces_dpd();
    if (wall0) forces_wall();
    forces_rbc();

    if (contactforces) forces_cnt(&w_r);
    forces_fsi(&w_s, &w_r);

    rex::bind_solutes(w_r);
    rex::pack_p();
    rex::post_p();
    rex::recv_p();

    rex::halo(); /* fsi::halo(); */

    rex::post_f();
    rex::recv_f();

    dSync();
}

void dev2hst() { /* device to host  data transfer */
    int start = 0;
    cD2H(a::pp_hst + start, o::pp, o::n); start += o::n;
    if (solids0) {
        cD2H(a::pp_hst + start, s::q.pp, s::q.n); start += s::q.n;
    }
    if (rbcs) {
        cD2H(a::pp_hst + start, r::q.pp, r::q.n); start += r::q.n;
    }
}

void dump_part(int step) {
        cD2H(o::pp_hst, o::pp, o::n);
        dump::parts(o::pp_hst, o::n, "solvent", step);

        if(solids0) {
            cD2H(s::q.pp_hst, s::q.pp, s::q.n);
            dump::parts(s::q.pp_hst, s::q.n, "solid", step);
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
  cD2H(a::pp_hst, o::pp, o::n);
  dump_field->dump(a::pp_hst, o::n);
}

void diag(int it) {
    int n = o::n + s::q.n + r::q.n; dev2hst();
    diagnostics(a::pp_hst, n, it);
}

void body_force(float driving_force0) {
    k_sim::body_force<<<k_cnf(o::n)>>> (1, o::pp, o::ff, o::n, driving_force0);

    if (solids0 && s::q.n)
    k_sim::body_force<<<k_cnf(s::q.n)>>> (solid_mass, s::q.pp, s::ff, s::q.n, driving_force0);

    if (rbcs && r::q.n)
    k_sim::body_force<<<k_cnf(r::q.n)>>> (rbc_mass, r::q.pp, r::ff, r::q.n, driving_force0);
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

void dump_diag_after(int it) { /* after wall */
    if (it % part_freq)
    solid::dump(it, s::q.ss_dmp, s::t.ss_dmp, s::q.ns, m::coords);
}

void dump_diag0(int it) { /* generic dump */
    if (it % part_freq  == 0) {
        if (part_dumps) dump_part(it);
        dump_rbcs();
        diag(it);
    }
    if (field_dumps && it % field_freq == 0) dump_grid();
}

void dump_diag(int it, bool wall0) { /* dump and diag */
    dump_diag0(it);
    if (wall0) dump_diag_after(it);
}

void step(float driving_force0, bool wall0, int it) {
    assert(o::n <= MAX_PART_NUM);
    assert(r::q.n <= MAX_PART_NUM);
    flu::distr(&o::pp, &o::n, o::cells, &o::td, &o::tz, &o::w);
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
    set_ids_solids();
    if (solids0 && s::q.n) k_sim::clear_velocity<<<k_cnf(s::q.n)>>>(s::q.pp, s::q.n);
    if (rbcs    && r::q.n) k_sim::clear_velocity<<<k_cnf(r::q.n)>>>(r::q.pp, r::q.n);
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
    dpd::fin();
    dump::fin();
    rex::fin();
    fsi::fin();

    if (solids0) mrescue::fin();

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
