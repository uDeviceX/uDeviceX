#define OO dbg::check_vv(o::q.pp, o::q.n, F("")), dbg::check_ff(o::ff, o::q.n, F("A")), dbg::check_pos_pu(o::q.pp, o::q.n, F(""))
#define SS dbg::check_vv(s::q.pp, s::q.n, F("")), dbg::check_ff(s::ff, s::q.n, F("A")), dbg::check_pos_pu(s::q.pp, s::q.n, F(""))
#define SYNC dSync(); MC(l::m::Barrier(l::m::cart));

void forces(bool wall0) {
    clear_forces(o::ff, o::q.n);
    if (solids0) clear_forces(s::ff, s::q.n);
    if (rbcs)    clear_forces(r::ff, r::q.n);

    OO;
    SS;
    forces_dpd();
    OO;
    SS;

    if (wall0 && w::q.n) forces_wall();
    OO;
    SS;

    forces_rbc();

    std::vector<ParticlesWrap> w_r;
    if (solids0) w_r.push_back(ParticlesWrap(s::q.pp, s::q.n, s::ff));
    if (rbcs)    w_r.push_back(ParticlesWrap(r::q.pp, r::q.n, r::ff));
    OO;
    SS;
    if (contactforces) forces_cnt(&w_r);
    OO;
    SS;
    SolventWrap w_s(o::q.pp, o::q.n, o::ff, o::q.cells->start, o::q.cells->count);
    if (fsiforces)     forces_fsi(&w_s, &w_r);
    OO;
    SS;

    x::rex(w_r); /* fsi::halo(), cnt::halo() */
    OO;
    SS;

    dSync();
}
