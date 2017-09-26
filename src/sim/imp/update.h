void clear_vel() {
    scheme::clear_vel(o::q.pp, o::q.n);
    if (solids) scheme::clear_vel(s::q.pp, s::q.n);
    if (rbcs  ) scheme::clear_vel(r::q.pp, r::q.n);
}

void update_solid() {
    if (!s::q.n) return;

    rig::update(s::ff, s::q.rr0, s::q.n, s::q.ns, /**/ s::q.pp, s::q.ss);
    rig::update_mesh(s::q.ss, s::q.ns, s::q.nv, s::q.dvv, /**/ s::q.i_pp);
    // for dump
    cD2H(s::q.ss_dmp, s::q.ss, s::q.ns);
    rig::reinit_ft(s::q.ns, /**/ s::q.ss);
}

void bounce_solid(long it) {
#define DEV (false)
    mesh::get_bboxes_dev(s::q.i_pp, s::q.nv, s::q.ns, /**/ s::t.minbb_dev, s::t.maxbb_dev);

    cD2H(s::t.minbb_hst, s::t.minbb_dev, s::q.ns);
    cD2H(s::t.maxbb_hst, s::t.maxbb_dev, s::q.ns);
    cD2H(s::q.ss_hst, s::q.ss, s::q.ns);

    /* exchange solid meshes with neighbours */

    bbhalo::pack_sendcnt <DEV> (s::q.ss_hst, s::q.ns, s::q.i_pp, s::q.nv, s::t.minbb_hst, s::t.maxbb_hst);
    const int nsbb = bbhalo::post(s::q.nv);
    bbhalo::unpack <DEV> (s::q.nv, /**/ s::t.ss_hst, s::t.i_pp);

    cH2D(s::t.ss, s::t.ss_hst, nsbb);

    tcells::build_dev(s::q.nt, s::q.nv, s::q.dtt, s::t.i_pp, nsbb, /**/ &bb::qtc, /*w*/ &s::ws);

    mbounce::bounce_dev(o::ff, s::q.nt, s::q.nv, s::q.dtt, s::t.i_pp, bb::qtc.ss_dev, bb::qtc.cc_dev, bb::qtc.ii_dev, o::q.n, nsbb*s::q.nt, /**/ o::q.pp, &bb::tm);
    mbounce::collect_rig_dev(s::q.nt, nsbb, &bb::tm, /**/ s::t.ss);

    if (it % rescue_freq == 0)
        mrescue::rescue_dev(s::q.nt, s::q.nv, s::q.dtt, s::t.i_pp, nsbb, o::q.n, bb::qtc.ss_dev, bb::qtc.cc_dev, bb::qtc.ii_dev, /**/ o::q.pp);

    // send back fo, to

    cD2H(s::t.ss_hst, s::t.ss, nsbb);

    bbhalo::pack_back(s::t.ss_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::q.ss_hst);

    cH2D(s::q.ss, s::q.ss_hst, s::q.ns);

    // for dump
    memcpy(s::t.ss_dmp, s::q.ss_hst, s::q.ns * sizeof(Solid));
#undef DEV
}

void bounce_solid_new(long it) {
#define DEV (false)
    mesh::get_bboxes_dev(s::q.i_pp, s::q.nv, s::q.ns, /**/ s::t.minbb_dev, s::t.maxbb_dev);

    cD2H(s::t.minbb_hst, s::t.minbb_dev, s::q.ns);
    cD2H(s::t.maxbb_hst, s::t.maxbb_dev, s::q.ns);
    cD2H(s::q.ss_hst, s::q.ss, s::q.ns);

    /* exchange solid meshes with neighbours */

    bbhalo::pack_sendcnt <DEV> (s::q.ss_hst, s::q.ns, s::q.i_pp, s::q.nv, s::t.minbb_hst, s::t.maxbb_hst);
    const int nsbb = bbhalo::post(s::q.nv);
    bbhalo::unpack <DEV> (s::q.nv, /**/ s::t.ss_hst, s::t.i_pp);

    cH2D(s::t.ss, s::t.ss_hst, nsbb);
    
    /* new part */
    int n, nm, nt, *ss, *cc;
    int4 *tt;
    Particle *pp, *i_pp;
    int3 L = make_int3(XS, YS, ZS);
    
    nm = s::q.ns;
    nt = s::q.nt;
    tt = s::q.dtt;
    i_pp = s::t.i_pp;

    n  = o::q.n;
    pp = o::q.pp;
    cc = o::q.cells.counts;
    ss = o::q.cells.starts;
    
    meshbb::reini(n, /**/ bb::bbd);
    meshbb::find_collisions(nm, nt, tt, i_pp, L, ss, cc, pp, o::ff, /**/ bb::bbd);
    meshbb::select_collisions(n, /**/ bb::bbd);
    meshbb::bounce(n, bb::bbd, o::ff, nt, tt, i_pp, /**/ pp, bb::mm);

    /* end new part  */

    if (it % rescue_freq == 0)
        mrescue::rescue_dev(s::q.nt, s::q.nv, s::q.dtt, s::t.i_pp, nsbb, o::q.n, bb::qtc.ss_dev, bb::qtc.cc_dev, bb::qtc.ii_dev, /**/ o::q.pp);

    // send back fo, to

    cD2H(s::t.ss_hst, s::t.ss, nsbb);

    bbhalo::pack_back(s::t.ss_hst);
    bbhalo::post_back();
    bbhalo::unpack_back(s::q.ss_hst);

    cH2D(s::q.ss, s::q.ss_hst, s::q.ns);

    // for dump
    memcpy(s::t.ss_dmp, s::q.ss_hst, s::q.ns * sizeof(Solid));
#undef DEV
}

void update_solvent(long it) {
    scheme::restrain(o::qc.ii, o::q.n, it, /**/ o::q.pp);
    scheme::move(dpd_mass, o::q.pp, o::ff, o::q.n);
}

void update_rbc(long it) {
    bool cond;
    cond = multi_solvent && color_freq && it % color_freq == 0;
    if (cond) {MSG("recolor"); gen_colors();};
    scheme::move(rbc_mass, r::q.pp, r::ff, r::q.n);
}

void bounce() {
    sdf::bounce(&w::qsdf, o::q.n, /**/ o::q.pp);
    // if (rbcs) sdf::bounce(&w::qsdf, r::q.n, /**/ r::q.pp);
}

/* single node only for now */
void bounce_rbc() {
    // TODO
    // build_tcells_dev(s::q.m_dev, s::t.i_pp, nsbb, /**/ s::t.tcs_dev, s::t.tcc_dev, s::t.tci_dev, /*w*/ &s::ws);
    // mbounce::bounce_dev(o::ff, s::q.m_dev, s::t.i_pp, s::t.tcs_dev, s::t.tcc_dev, s::t.tci_dev, o::q.n, nsbb*s::q.m_dev.nt, /**/ o::q.pp, &bb::tm);
    // mbounce::collect_rbc_dev(s::q.m_dev.nt, nsbb, &bb::tm, /**/ s::t.ss);
}
