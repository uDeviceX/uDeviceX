namespace sim {
void distr_solid() {
    if (s::ns) cD2H(s::ss_hst, s::ss_dev, s::ns);
    sdstr::pack_sendcnt <DEV> (s::ss_hst, s::i_pp_dev, s::ns, s::m_dev.nv);
    s::ns = sdstr::post(s::m_dev.nv);
    s::npp = s::ns * s::nps;
    sdstr::unpack <DEV> (s::m_dev.nv, /**/ s::ss_hst, s::i_pp_dev);
    if (s::ns) cH2D(s::ss_dev, s::ss_hst, s::ns);
    solid::generate_dev(s::ss_dev, s::ns, s::rr0, s::nps, /**/ s::pp);
}

void update_solid0() {
    solid::update_dev(s::ff, s::rr0, s::npp, s::ns, /**/ s::pp, s::ss_dev);
    solid::update_mesh_dev(s::ss_dev, s::ns, s::m_dev, /**/ s::i_pp_dev);

    // for dump
    cD2H(s::ss_dmphst, s::ss_dev, s::ns);

    solid::reinit_ft_dev(s::ns, /**/ s::ss_dev);
}

void bounce_solid(int it) {
    collision::get_bboxes_dev(s::i_pp_dev, s::m_dev.nv, s::ns, /**/ s::minbb_dev, s::maxbb_dev);

    cD2H(s::minbb_hst, s::minbb_dev, s::ns);
    cD2H(s::maxbb_hst, s::maxbb_dev, s::ns);
    cD2H(s::ss_hst, s::ss_dev, s::ns);

    /* exchange solid meshes with neighbours */

    bbhalo::pack_sendcnt <DEV> (s::ss_hst, s::ns, s::i_pp_dev, s::m_dev.nv, s::minbb_hst, s::maxbb_hst);
    const int nsbb = bbhalo::post(s::m_dev.nv);
    bbhalo::unpack <DEV> (s::m_dev.nv, /**/ s::ss_bb_hst, s::i_pp_bb_dev);

    cH2D(s::ss_bb_dev, s::ss_bb_hst, nsbb);

    build_tcells_dev(s::m_dev, s::i_pp_bb_dev, nsbb, /**/ s::tcs_dev, s::tcc_dev, s::tci_dev);

    mbounce::bounce_tcells_dev(o::ff, s::m_dev, s::i_pp_bb_dev, s::tcs_dev, s::tcc_dev, s::tci_dev, o::n, /**/ o::pp, s::ss_bb_dev);

    if (it % rescue_freq == 0)
    mrescue::rescue_dev(s::m_dev, s::i_pp_bb_dev, nsbb, o::n, s::tcs_dev, s::tcc_dev, s::tci_dev, /**/ o::pp);

    // send back fo, to

    cD2H(s::ss_bb_hst, s::ss_bb_dev, nsbb);

    bbhalo::pack_back(s::ss_bb_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::ss_hst);

    cH2D(s::ss_dev, s::ss_hst, s::ns);

    // for dump
    memcpy(s::ss_dmpbbhst, s::ss_hst, s::ns * sizeof(Solid));
}
  
} /* namespace sim */
