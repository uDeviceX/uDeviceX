namespace sim {
void distr_solid() {
    if (s::q.ns) cD2H(s::q.ss_hst, s::q.ss, s::q.ns);
    sdstr::pack_sendcnt <DEV> (s::q.ss_hst, s::q.i_pp, s::q.ns, s::q.m_dev.nv);
    s::q.ns = sdstr::post(s::q.m_dev.nv);
    s::q.n = s::q.ns * s::q.nps;
    sdstr::unpack <DEV> (s::q.m_dev.nv, /**/ s::q.ss_hst, s::q.i_pp);
    if (s::q.ns) cH2D(s::q.ss, s::q.ss, s::q.ns);
    solid::generate_dev(s::q.ss, s::q.ns, s::q.rr0, s::q.nps, /**/ s::q.pp);
}

void update_solid0() {
    solid::update_dev(s::ff, s::q.rr0, s::q.n, s::q.ns, /**/ s::q.pp, s::q.ss);
    solid::update_mesh_dev(s::q.ss, s::q.ns, s::q.m_dev, /**/ s::q.i_pp);

    // for dump
    cD2H(s::q.ss_dmp, s::q.ss, s::q.ns);

    solid::reinit_ft_dev(s::q.ns, /**/ s::q.ss);
}

void bounce_solid(int it) {
    collision::get_bboxes_dev(s::q.i_pp, s::q.m_dev.nv, s::q.ns, /**/ s::t.minbb_dev, s::t.maxbb_dev);

    cD2H(s::t.minbb_hst, s::t.minbb_dev, s::q.ns);
    cD2H(s::t.maxbb_hst, s::t.maxbb_dev, s::q.ns);
    cD2H(s::q.ss_hst, s::q.ss, s::q.ns);

    /* exchange solid meshes with neighbours */

    bbhalo::pack_sendcnt <DEV> (s::q.ss_hst, s::q.ns, s::q.i_pp, s::q.m_dev.nv, s::t.minbb_hst, s::t.maxbb_hst);
    const int nsbb = bbhalo::post(s::q.m_dev.nv);
    bbhalo::unpack <DEV> (s::q.m_dev.nv, /**/ s::t.ss_hst, s::t.i_pp);

    cH2D(s::t.ss, s::t.ss_hst, nsbb);

    build_tcells_dev(s::q.m_dev, s::t.i_pp, nsbb, /**/ s::t.tcs_dev, s::t.tcc_dev, s::t.tci_dev);

    mbounce::bounce_tcells_dev(o::ff, s::q.m_dev, s::t.i_pp, s::t.tcs_dev, s::t.tcc_dev, s::t.tci_dev, o::n, /**/ o::pp, s::t.ss);

    if (it % rescue_freq == 0)
    mrescue::rescue_dev(s::q.m_dev, s::t.i_pp, nsbb, o::n, s::t.tcs_dev, s::t.tcc_dev, s::t.tci_dev, /**/ o::pp);

    // send back fo, to

    cD2H(s::t.ss_hst, s::t.ss, nsbb);

    bbhalo::pack_back(s::t.ss_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::q.ss_hst);

    cH2D(s::q.ss, s::q.ss_hst, s::q.ns);

    // for dump
    memcpy(s::t.ss_dmp, s::q.ss_hst, s::q.ns * sizeof(Solid));
}
  
} /* namespace sim */
