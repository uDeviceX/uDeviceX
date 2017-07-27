void update_solid() {
    if (s::q.n) update_solid0();
}

void update_solvent() {
    if (o::q.n) dev::update<<<k_cnf(o::q.n)>>> (dpd_mass, o::q.pp, o::ff, o::q.n);
}

void update_rbc() {
    if (r::q.n) dev::update<<<k_cnf(r::q.n)>>> (rbc_mass, r::q.pp, r::ff, r::q.n);
}

void bounce() {
    sdf::bounce(w::qsdf.texsdf, o::q.n, /**/ o::q.pp);
    // if (rbcs) sdf::bounce(w::qsdf.texsdf, r::q.n, /**/ r::q.pp);
}
