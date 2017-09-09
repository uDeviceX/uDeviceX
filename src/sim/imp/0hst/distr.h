void distr_solid() {
    sdstr::pack_sendcnt <HST> (s::q.ss_hst, s::q.i_pp_hst, s::q.ns, s::q.m_hst.nv);
    s::q.ns = sdstr::post(s::q.m_hst.nv);
    s::q.n = s::q.ns * s::q.nps;
    sdstr::unpack <HST> (s::q.m_hst.nv, /**/ s::q.ss_hst, s::q.i_pp_hst);
    rig::generate(s::q.ss_hst, s::q.ns, s::q.rr0_hst, s::q.nps, /**/ s::q.pp_hst);
    cH2D(s::q.pp, s::q.pp_hst, 3 * s::q.n);
}
