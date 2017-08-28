#define O  OP, OF
#define OP dbg::check_vv(o::q.pp, o::q.n, F("vel")), dbg::check_pos_pu(o::q.pp, o::q.n, F("pos"))
#define OF dbg::check_ff(o::ff, o::q.n, F("ff"))

#define S  OP, OF
#define SP dbg::check_vv(s::q.pp, s::q.n, F("vel")), dbg::check_pos_pu(s::q.pp, s::q.n, F("pos"))
#define SF dbg::check_ff(s::ff, s::q.n, F("ff"))

#define SYNC dSync(); MC(l::m::Barrier(l::m::cart));

void forces(bool wall0) {
    clear_forces(o::ff, o::q.n);
    if (solids0) clear_forces(s::ff, s::q.n);
    if (rbcs)    clear_forces(r::ff, r::q.n);

    O;
    S;
    forces_dpd();
    O;
    S;

    if (wall0 && w::q.n) forces_wall();
    O;
    S;

    forces_rbc();

    std::vector<ParticlesWrap> w_r;
    if (solids0) w_r.push_back(ParticlesWrap(s::q.pp, s::q.n, s::ff));
    if (rbcs)    w_r.push_back(ParticlesWrap(r::q.pp, r::q.n, r::ff));
    O;
    S;
    if (contactforces) forces_cnt(&w_r);
    O;
    S;
    SolventWrap w_s(o::q.pp, o::q.n, o::ff, o::q.cells->start, o::q.cells->count);
    if (fsiforces)     forces_fsi(&w_s, &w_r);
    O;
    S;

    x::rex(w_r); /* fsi::halo(), cnt::halo() */
    O;
    S;

    dSync();
}
