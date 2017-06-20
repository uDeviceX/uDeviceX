namespace sim {
void forces_rbc() {
    if (rbcs) rbc::forces(r::q, r::tt, /**/ r::ff);
}

void forces_dpd() {
    dpd::pack(o::pp, o::n, o::cells->start, o::cells->count);
    /* :TODO: breaks a contract with hiwi */
    dpd::flocal(o::tz.zip0, o::tz.zip1, o::n, o::cells->start, o::cells->count, /**/ o::ff);
    dpd::post(o::pp, o::n);
    dpd::recv();
    dpd::fremote(o::n, o::ff);
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
}
