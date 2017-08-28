void update_solid0() {
    solid::update_dev(s::ff, s::q.rr0, s::q.n, s::q.ns, /**/ s::q.pp, s::q.ss);
    solid::update_mesh_dev(s::q.ss, s::q.ns, s::q.m_dev, /**/ s::q.i_pp);

    // for dump
    cD2H(s::q.ss_dmp, s::q.ss, s::q.ns);

    solid::reinit_ft_dev(s::q.ns, /**/ s::q.ss);
}
