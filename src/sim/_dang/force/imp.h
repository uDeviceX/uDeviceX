void forces(bool wall0) {
    clear_forces(o::ff, o::q.n);
    if (solids0) clear_forces(s::ff, s::q.n);
    if (rbcs)    clear_forces(r::ff, r::q.n);

    forces_dpd();
    if (wall0 && w::q.n) forces_wall();
    forces_rbc();

    std::vector<ParticlesWrap> w_r;
    if (solids0) w_r.push_back(ParticlesWrap(s::q.pp, s::q.n, s::ff));
    if (rbcs)    w_r.push_back(ParticlesWrap(r::q.pp, r::q.n, r::ff));
    if (contactforces) forces_cnt(&w_r);

    SolventWrap w_s;
    w_s.pp = o::q.pp;
    w_s.ff = o::ff;
    w_s.n  = o::q.n;
    w_s.starts = o::q.cells->start;
    if (fsiforces)     forces_fsi(&w_s, &w_r);

    rex::rex(w_r); /* fsi::halo(), cnt::halo() */
    dSync();
}
