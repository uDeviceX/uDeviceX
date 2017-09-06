namespace scheme {
void move(float mass, Particle *pp, Force *ff, int n) {
    KL(dev::update, (k_cnf(n)), (mass, pp, ff, n));
}

void clear_vel(Particle *pp, int n) {
    KL(dev::clear_vel, (k_cnf(n)), (pp, n));
}

} /* namespace */

void clear_vel() {
    scheme::clear_vel(o::q.pp, o::q.n);
    if (solids) scheme::clear_vel(s::q.pp, s::q.n);
    if (rbcs  ) scheme::clear_vel(r::q.pp, r::q.n);
}

void update_solid() {
    if (s::q.n) update_solid0();
}

void update_solvent() {
    scheme::move(dpd_mass, o::q.pp, o::ff, o::q.n);
}

void update_rbc() {
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
