void update_solid0() {
    rig::update(s::ff, s::q.rr0, s::q.n, s::q.ns, /**/ s::q.pp, s::q.ss);
    rig::update_mesh(s::q.ss, s::q.ns, s::q.m_dev, /**/ s::q.i_pp);

    // for dump
    cD2H(s::q.ss_dmp, s::q.ss, s::q.ns);

    rig::reinit_ft(s::q.ns, /**/ s::q.ss);
}
