void bounce_solid(int it) {
    mesh::get_bboxes_dev(s::q.i_pp, s::q.m_dev.nv, s::q.ns, /**/ s::t.minbb_dev, s::t.maxbb_dev);

    cD2H(s::t.minbb_hst, s::t.minbb_dev, s::q.ns);
    cD2H(s::t.maxbb_hst, s::t.maxbb_dev, s::q.ns);
    cD2H(s::q.ss_hst, s::q.ss, s::q.ns);

    /* exchange solid meshes with neighbours */

    bbhalo::pack_sendcnt <DEV> (s::q.ss_hst, s::q.ns, s::q.i_pp, s::q.m_dev.nv, s::t.minbb_hst, s::t.maxbb_hst);
    const int nsbb = bbhalo::post(s::q.m_dev.nv);
    bbhalo::unpack <DEV> (s::q.m_dev.nv, /**/ s::t.ss_hst, s::t.i_pp);

    cH2D(s::t.ss, s::t.ss_hst, nsbb);

    tcells::build_dev(s::q.m_dev, s::t.i_pp, nsbb, /**/ &bb::qtc, /*w*/ &s::ws);

    mbounce::bounce_dev(o::ff, s::q.m_dev, s::t.i_pp, bb::qtc.ss_dev, bb::qtc.cc_dev, bb::qtc.ii_dev, o::q.n, nsbb*s::q.m_dev.nt, /**/ o::q.pp, &bb::tm);
    mbounce::collect_rig_dev(s::q.m_dev.nt, nsbb, &bb::tm, /**/ s::t.ss);

    if (it % rescue_freq == 0)
    mrescue::rescue_dev(s::q.m_dev, s::t.i_pp, nsbb, o::q.n, bb::qtc.ss_dev, bb::qtc.cc_dev, bb::qtc.ii_dev, /**/ o::q.pp);

    // send back fo, to

    cD2H(s::t.ss_hst, s::t.ss, nsbb);

    bbhalo::pack_back(s::t.ss_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::q.ss_hst);

    cH2D(s::q.ss, s::q.ss_hst, s::q.ns);

    // for dump
    memcpy(s::t.ss_dmp, s::q.ss_hst, s::q.ns * sizeof(Solid));
}
