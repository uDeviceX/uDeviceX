void clear_vel() {
    KL(dev::clear_vel, (k_cnf(o::q.n)), (o::q.pp, o::q.n));
    if (solids) KL(dev::clear_vel, (k_cnf(s::q.n)), (s::q.pp, s::q.n));
    if (rbcs  ) KL(dev::clear_vel, (k_cnf(r::q.n)), (r::q.pp, r::q.n));
}

void update_solid() {
    if (s::q.n)
        update_solid0();
}

void update_solvent() {
    using namespace o;
    KL(dev::update, (k_cnf(q.n)), (dpd_mass, q.pp, ff, q.n));
}

void update_rbc() {
    dbg::check_pp_pu(r::q.pp, r::q.n, "rbc, before");
    KL(dev::update, (k_cnf(r::q.n)),  (rbc_mass, r::q.pp, r::ff, r::q.n));
    dbg::check_pp_pu(r::q.pp, r::q.n, "rbc, update");
}

void bounce() {
    dbg::check_pp_pu(o::q.pp, o::q.n, "flu, before, bounce-back");
    sdf::bounce(&w::qsdf, o::q.n, /**/ o::q.pp);
    // if (rbcs) sdf::bounce(&w::qsdf, r::q.n, /**/ r::q.pp);
    dbg::check_pp_pu(o::q.pp, o::q.n, "flu, after, bounce-back");
}

void bounce_rbc() { }
