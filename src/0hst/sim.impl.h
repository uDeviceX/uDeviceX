void distr_solid() {
    sdstr::pack_sendcnt <HST> (s::q.ss_hst, s::q.i_pp_hst, s::q.ns, s::q.m_hst.nv);
    s::q.ns = sdstr::post(s::q.m_hst.nv);
    s::q.n = s::q.ns * s::q.nps;
    sdstr::unpack <HST> (s::q.m_hst.nv, /**/ s::q.ss_hst, s::q.i_pp_hst);
    solid::generate_hst(s::q.ss_hst, s::q.ns, s::q.rr0_hst, s::q.nps, /**/ s::q.pp_hst);
    cH2D(s::q.pp, s::q.pp_hst, 3 * s::q.n);
}

void update_solid0() {
    cD2H(s::q.pp_hst, s::q.pp, s::q.n);
    cD2H(s::ff_hst, s::ff, s::q.n);
  
    solid::update_hst(s::ff_hst, s::q.rr0_hst, s::q.n, s::q.ns, /**/ s::q.pp_hst, s::q.ss_hst);
    solid::update_mesh_hst(s::q.ss_hst, s::q.ns, s::q.m_hst, /**/ s::q.i_pp_hst);
  
    // for dump
    memcpy(s::q.ss_dmp, s::q.ss_hst, s::q.ns * sizeof(Solid));
  
    solid::reinit_ft_hst(s::q.ns, /**/ s::q.ss_hst);
  
    cH2D(s::q.pp, s::q.pp_hst, s::q.n);
}

void bounce_solid(int it) {
    collision::get_bboxes_hst(s::q.i_pp_hst, s::q.m_hst.nv, s::q.ns, /**/ s::t.minbb_hst, s::t.maxbb_hst);

    /* exchange solid meshes with neighbours */

    bbhalo::pack_sendcnt <HST> (s::q.ss_hst, s::q.ns, s::q.i_pp_hst, s::q.m_hst.nv, s::t.minbb_hst, s::t.maxbb_hst);
    const int nsbb = bbhalo::post(s::q.m_hst.nv);
    bbhalo::unpack <HST> (s::q.m_hst.nv, /**/ s::t.ss_hst, s::t.i_pp_hst);

    build_tcells_hst(s::q.m_hst, s::t.i_pp_hst, nsbb, /**/ s::t.tcs_hst, s::t.tcc_hst, s::t.tci_hst);

    cD2H(o::q.pp_hst, o::q.pp, o::q.n);
    cD2H(o::ff_hst, o::ff, o::q.n);

    mbounce::bounce_tcells_hst(o::ff_hst, s::q.m_hst, s::t.i_pp_hst, s::t.tcs_hst, s::t.tcc_hst, s::t.tci_hst, o::q.n, /**/ o::q.pp_hst, s::t.ss_hst);

    if (it % rescue_freq == 0)
        mrescue::rescue_hst(s::q.m_hst, s::t.i_pp_hst, nsbb, o::q.n, s::t.tcs_hst, s::t.tcc_hst, s::t.tci_hst, /**/ o::q.pp_hst);

    cH2D(o::q.pp, o::q.pp_hst, o::q.n);

    // send back fo, to

    bbhalo::pack_back(s::t.ss_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::q.ss_hst);

    // for dump
    memcpy(s::q.ss_dmp, s::q.ss_hst, s::q.ns * sizeof(Solid));
}
