enum {NTHRD=128};

void rig_reinit_ft(const int nsolid, /**/ Solid *ss) {
    KL(dev::reinit_ft, (k_cnf(nsolid)), (nsolid, /**/ ss));
}

void rig_update(int n, const Force *ff, const float *rr0, int ns, /**/ Particle *pp, Solid *ss) {
    if (ns < 1) return;
        
    const int nps = n / ns; /* number of particles per solid */

    const dim3 nblck ( ceiln(nps, NTHRD), ns );
    const dim3 nthrd ( NTHRD, 1 );
        
    KL(dev::add_f_to, ( nblck, nthrd ), (nps, pp, ff, /**/ ss));
    KL(dev::update_om_v, (1, ns), (ns, /**/ ss));
    if (!pin_com) KL(dev::update_com, (1, 3*ns ), (ns, /**/ ss));        
    KL(dev::rot_referential,(1, ns), (ns, /**/ ss));

    KL(dev::update_pp, ( nblck, nthrd ), (nps, rr0, ss, /**/ pp));
}

void rig_generate(int ns, const Solid *ss, int nps, const float *rr0, /**/ Particle *pp) {
    if (ns < 1) return;
        
    const dim3 nblck ( ceiln(nps, NTHRD), ns );
    const dim3 nthrd ( NTHRD, 1 );

    KL(dev::update_pp, ( nblck, nthrd ), (nps, rr0, ss, /**/ pp));
}

void rig_update_mesh(int ns, const Solid *ss, int nv, const float *vv, /**/ Particle *pp) {
    const dim3 nblck ( ceiln(nv, NTHRD), ns );
    const dim3 nthrd ( NTHRD, 1 );

    KL(dev::update_mesh, ( nblck, nthrd ), (ss, nv, vv, /**/ pp));
}
