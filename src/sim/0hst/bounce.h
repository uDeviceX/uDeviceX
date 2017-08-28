
    mesh::get_bboxes_hst(s::q.i_pp_hst, s::q.m_hst.nv, s::q.ns, /**/ s::t.minbb_hst, s::t.maxbb_hst);

    /* exchange solid meshes with neighbours */

    bbhalo::pack_sendcnt <HST> (s::q.ss_hst, s::q.ns, s::q.i_pp_hst, s::q.m_hst.nv, s::t.minbb_hst, s::t.maxbb_hst);
    const int nsbb = bbhalo::post(s::q.m_hst.nv);
    bbhalo::unpack <HST> (s::q.m_hst.nv, /**/ s::t.ss_hst, s::t.i_pp_hst);

    tcells::build_hst(s::q.m_hst, s::t.i_pp_hst, nsbb, /**/ &bb::qtc);

    cD2H(o::q.pp_hst, o::q.pp, o::q.n);
    cD2H(o::ff_hst, o::ff, o::q.n);

    mbounce::bounce_hst(o::ff_hst, s::q.m_hst, s::t.i_pp_hst, bb::qtc.ss_hst, bb::qtc.cc_hst, bb::qtc.ii_hst, o::q.n, nsbb*s::q.m_dev.nt, /**/ o::q.pp_hst, &bb::tm);
    mbounce::collect_rig_dev(s::q.m_hst.nt, nsbb, &bb::tm, /**/ s::t.ss_hst);

    if (it % rescue_freq == 0)
        mrescue::rescue_hst(s::q.m_hst, s::t.i_pp_hst, nsbb, o::q.n, bb::qtc.ss_hst, bb::qtc.cc_hst, bb::qtc.ii_hst, /**/ o::q.pp_hst);

    cH2D(o::q.pp, o::q.pp_hst, o::q.n);

    // send back fo, to

    bbhalo::pack_back(s::t.ss_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::q.ss_hst);

    // for dump
    memcpy(s::q.ss_dmp, s::q.ss_hst, s::q.ns * sizeof(Solid));
}
