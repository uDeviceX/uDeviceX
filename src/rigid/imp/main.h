void reinit_ft(const int nsolid, /**/ Solid *ss) {
    KL(dev::reinit_ft, (k_cnf(nsolid)), (nsolid, /**/ ss));
}
    
void update(const Force *ff, const float *rr0, int n, int ns, /**/ Particle *pp, Solid *ss) {
    if (ns < 1) return;
        
    const int nps = n / ns; /* number of particles per solid */

    const dim3 nblck ( (127 + nps) / 128, ns );
    const dim3 nthrd ( 128, 1 );
        
    KL(dev::add_f_to, ( nblck, nthrd ), (nps, pp, ff, /**/ ss));
    KL(dev::update_om_v, (1, ns), (ns, /**/ ss));
    KL(dev::compute_velocity, ( nblck, nthrd ), (nps, ss, /**/ pp));

    if (!pin_com) KL(dev::update_com, (1, 3*ns ), (ns, /**/ ss));
        
    KL(dev::rot_referential,(1, ns), (ns, /**/ ss));

    KL(dev::update_r, ( nblck, nthrd ), (nps, rr0, ss, /**/ pp));
}

void generate(const Solid *ss, const int ns, const float *rr0, const int nps, /**/ Particle *pp) {
    if (ns < 1) return;
        
    const dim3 nblck ( (127 + nps) / 128, ns );
    const dim3 nthrd ( 128, 1 );

    KL(dev::update_r, ( nblck, nthrd ), (nps, rr0, ss, /**/ pp));
    KL(dev::compute_velocity, ( nblck, nthrd ), (nps, ss, /**/ pp));
}

void update_mesh(const Solid *ss, const int ns, int nv, const float *vv, /**/ Particle *pp) {
    const dim3 nthrd(128, 1);
    const dim3 nblck((nv + 127)/128, ns);

    KL(dev::update_mesh, ( nblck, nthrd ), (ss, nv, vv, /**/ pp));
}
