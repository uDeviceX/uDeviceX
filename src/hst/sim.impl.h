namespace sim {
void distr_solid() {
#ifdef DEVICE_SOLID
    if (s::ns) cD2H(s::ss_hst, s::ss_dev, s::ns);
    sdstr::pack_sendcnt <DEV> (s::ss_hst, s::i_pp_dev, s::ns, s::m_dev.nv);
    s::ns = sdstr::post(s::m_dev.nv);
    s::npp = s::ns * s::nps;
    sdstr::unpack <DEV> (s::m_dev.nv, /**/ s::ss_hst, s::i_pp_dev);
    if (s::ns) cH2D(s::ss_dev, s::ss_hst, s::ns);
    solid::generate_dev(s::ss_dev, s::ns, s::rr0, s::nps, /**/ s::pp);
#else
    sdstr::pack_sendcnt <HST> (s::ss_hst, s::i_pp_hst, s::ns, s::m_hst.nv);
    s::ns = sdstr::post(s::m_hst.nv);
    s::npp = s::ns * s::nps;
    sdstr::unpack <HST> (s::m_hst.nv, /**/ s::ss_hst, s::i_pp_hst);
    solid::generate_hst(s::ss_hst, s::ns, s::rr0_hst, s::nps, /**/ s::pp_hst);
    cH2D(s::pp, s::pp_hst, 3 * s::npp);
#endif
}

  void update_solid0() {
#ifndef DEVICE_SOLID
    cD2H(s::pp_hst, s::pp, s::npp);
    cD2H(s::ff_hst, s::ff, s::npp);

    solid::update_hst(s::ff_hst, s::rr0_hst, s::npp, s::ns, /**/ s::pp_hst, s::ss_hst);
    solid::update_mesh_hst(s::ss_hst, s::ns, s::m_hst, /**/ s::i_pp_hst);

    // for dump
    memcpy(s::ss_dmphst, s::ss_hst, s::ns * sizeof(Solid));

    solid::reinit_ft_hst(s::ns, /**/ s::ss_hst);

    cH2D(s::pp, s::pp_hst, s::npp);
#else
    solid::update_dev(s::ff, s::rr0, s::npp, s::ns, /**/ s::pp, s::ss_dev);
    solid::update_mesh_dev(s::ss_dev, s::ns, s::m_dev, /**/ s::i_pp_dev);

    // for dump
    cD2H(s::ss_dmphst, s::ss_dev, s::ns);

    solid::reinit_ft_dev(s::ns, /**/ s::ss_dev);
#endif
}

void bounce_solid(int it) {
#ifndef DEVICE_SOLID

    collision::get_bboxes_hst(s::i_pp_hst, s::m_hst.nv, s::ns, /**/ s::bboxes_hst);

    /* exchange solid meshes with neighbours */

    bbhalo::pack_sendcnt <HST> (s::ss_hst, s::ns, s::i_pp_hst, s::m_hst.nv, s::bboxes_hst);
    const int nsbb = bbhalo::post(s::m_hst.nv);
    bbhalo::unpack <HST> (s::m_hst.nv, /**/ s::ss_bb_hst, s::i_pp_bb_hst);

    build_tcells_hst(s::m_hst, s::i_pp_bb_hst, nsbb, /**/ s::tcs_hst, s::tcc_hst, s::tci_hst);

    cD2H(o::pp_hst, o::pp, o::n);
    cD2H(o::ff_hst, o::ff, o::n);

    mbounce::bounce_tcells_hst(o::ff_hst, s::m_hst, s::i_pp_bb_hst, s::tcs_hst, s::tcc_hst, s::tci_hst, o::n, /**/ o::pp_hst, s::ss_bb_hst);

    if (it % rescue_freq == 0)
    mrescue::rescue_hst(s::m_hst, s::i_pp_bb_hst, nsbb, o::n, s::tcs_hst, s::tcc_hst, s::tci_hst, /**/ o::pp_hst);

    cH2D(o::pp, o::pp_hst, o::n);

    // send back fo, to

    bbhalo::pack_back(s::ss_bb_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::ss_hst);

    // for dump
    memcpy(s::ss_dmpbbhst, s::ss_hst, s::ns * sizeof(Solid));

#else // bounce on device

    collision::get_bboxes_dev(s::i_pp_dev, s::m_dev.nv, s::ns, /**/ s::bboxes_dev);

    cD2H(s::bboxes_hst, s::bboxes_dev, 6 * s::ns);
    cD2H(s::ss_hst, s::ss_dev, s::ns);

    /* exchange solid meshes with neighbours */

    bbhalo::pack_sendcnt <DEV> (s::ss_hst, s::ns, s::i_pp_dev, s::m_dev.nv, s::bboxes_hst);
    const int nsbb = bbhalo::post(s::m_dev.nv);
    bbhalo::unpack <DEV> (s::m_dev.nv, /**/ s::ss_bb_hst, s::i_pp_bb_dev);

    cH2D(s::ss_bb_dev, s::ss_bb_hst, nsbb);

    build_tcells_dev(s::m_dev, s::i_pp_bb_dev, s::ns, /**/ s::tcs_dev, s::tcc_dev, s::tci_dev);

    mbounce::bounce_tcells_dev(o::ff, s::m_dev, s::i_pp_bb_dev, s::tcs_dev, s::tcc_dev, s::tci_dev, o::n, /**/ o::pp, s::ss_dev);

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

#endif
}
  
} /* namespace sim */
