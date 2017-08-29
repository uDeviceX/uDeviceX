namespace rig
{

void reinit_ft(const int nsolid, /**/ Solid *ss) {
    KL(dev::reinit_ft, (k_cnf(nsolid)), (nsolid, /**/ ss));
}
    
void update(const Force *ff, const float *rr0, int n, int ns, /**/ Particle *pp, Solid *ss) {
    if (ns < 1) return;
        
    const int nps = n / ns; /* number of particles per solid */

    const dim3 nblck ( (127 + nps) / 128, ns );
    const dim3 nthrd ( 128, 1 );
        
    KL(dev::add_f_to, ( nblck, nthrd ), (pp, ff, nps, ns, /**/ ss));
    KL(dev::update_om_v, (1, ns), (ns, /**/ ss));
    KL(dev::compute_velocity, ( nblck, nthrd ), (ss, ns, nps, /**/ pp));

    if (!pin_com) KL(dev::update_com, (1, 3*ns ), (ns, /**/ ss));
        
    KL(dev::rot_referential,(1, ns), (ns, /**/ ss));

    KL(dev::update_r, ( nblck, nthrd ), (rr0, nps, ss, ns, /**/ pp));
}

void generate(const Solid *ss, const int ns, const float *rr0, const int nps, /**/ Particle *pp) {
    if (ns < 1) return;
        
    const dim3 nblck ( (127 + nps) / 128, ns );
    const dim3 nthrd ( 128, 1 );

    KL(dev::update_r, ( nblck, nthrd ), (rr0, nps, ss, ns, /**/ pp));
    KL(dev::compute_velocity, ( nblck, nthrd ), (ss, ns, nps, /**/ pp));
}

void update_mesh(const Solid *ss, const int ns, const Mesh m, /**/ Particle *pp) {
    const dim3 nthrd(128, 1);
    const dim3 nblck((m.nv + 127)/128, ns);

    KL(dev::update_mesh, ( nblck, nthrd ), (ss, m.vv, m.nv, /**/ pp));
}

} // rig
