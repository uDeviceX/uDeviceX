void distr_solid() {
    OO;
    SS;
    if (s::q.ns) cD2H(s::q.ss_hst, s::q.ss, s::q.ns);
    sdstr::pack_sendcnt <DEV> (s::q.ss_hst, s::q.i_pp, s::q.ns, s::q.m_dev.nv);
    s::q.ns = sdstr::post(s::q.m_dev.nv);
    s::q.n = s::q.ns * s::q.nps;
    sdstr::unpack <DEV> (s::q.m_dev.nv, /**/ s::q.ss_hst, s::q.i_pp);
    if (s::q.ns) cH2D(s::q.ss, s::q.ss_hst, s::q.ns);
    solid::generate_dev(s::q.ss, s::q.ns, s::q.rr0, s::q.nps, /**/ s::q.pp);
    OO;
    SS;
}
