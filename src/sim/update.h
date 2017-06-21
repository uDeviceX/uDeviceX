namespace sim {
void update_solid() {
    if (s::q.n) update_solid0();
}

void update_solvent() {
    float mass = 1;
    if (o::q.n) k_sim::update<<<k_cnf(o::q.n)>>> (mass, o::q.pp, o::ff, o::q.n);
}

void update_rbc() {
    float mass = rbc_mass;
    if (r::q.n) k_sim::update<<<k_cnf(r::q.n)>>> (mass, r::q.pp, r::ff, r::q.n);
}

void bounce() {
    if (o::q.n) k_sdf::bounce<<<k_cnf(o::q.n)>>>((float2*)o::q.pp, o::q.n);
    //if (rbcs && r::n) k_sdf::bounce<<<k_cnf(r::n)>>>((float2*)r::pp, r::n);
}
}
