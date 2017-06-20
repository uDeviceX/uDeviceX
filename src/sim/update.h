namespace sim {
void update_solid() {
    if (s::q.n) update_solid0();
}

void update_solvent() {
    float mass = 1;
    if (o::n) k_sim::update<<<k_cnf(o::n)>>> (mass, o::pp, o::ff, o::n);
}

void update_rbc() {
    float mass = rbc_mass;
    if (r::q.n) k_sim::update<<<k_cnf(r::q.n)>>> (mass, r::q.pp, r::ff, r::q.n);
}

void bounce() {
    if (o::n) k_sdf::bounce<<<k_cnf(o::n)>>>((float2*)o::pp, o::n);
    //if (rbcs && r::n) k_sdf::bounce<<<k_cnf(r::n)>>>((float2*)r::pp, r::n);
}
}
