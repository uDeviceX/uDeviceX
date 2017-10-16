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

void update_mesh(const Solid *ss, const int ns, int nv, const float *vv, /**/ Particle *pp) {
    const dim3 nthrd(128, 1);
    const dim3 nblck((nv + 127)/128, ns);

    KL(dev::update_mesh, ( nblck, nthrd ), (ss, vv, nv, /**/ pp));
}

#ifdef spdir // open geometry, use particles    
static void init_I_frompp(const Particle *pp, int n, float pmass, const float *com, /**/ float *I) {
    for (int c = 0; c < 6; ++c) I[c] = 0;

    for (int ip = 0; ip < n; ++ip) {
        const float *r0 = pp[ip].r;
        const float x = r0[X]-com[X], y = r0[Y]-com[Y], z = r0[Z]-com[Z];
        I[XX] += y*y + z*z;
        I[YY] += z*z + x*x;
        I[ZZ] += x*x + y*y;
        I[XY] -= x*y;
        I[XZ] -= z*x;
        I[YZ] -= y*z;
    }
    for (int c = 0; c < 6; ++c) I[c] *= pmass;
}
#else
static void init_I_fromm(float pmass, int nt, const int4 *tt, const float *vv, /**/ float *I) {
    float com[3] = {0};
    mesh::center_of_mass(nt, tt, vv, /**/ com);
    mesh::inertia_tensor(nt, tt, vv, com, numberdensity, /**/ I);

    for (int c = 0; c < 6; ++c) I[c] *= pmass;
}
#endif
    
void ini(const Particle *pp, int n, float pmass, const float *com, int nt, const int4 *tt, const float *vv, /**/ float *rr0, Solid *s) {
    s->v[X] = s->v[Y] = s->v[Z] = 0; 
    s->om[X] = s->om[Y] = s->om[Z] = 0; 

    /* ini basis vectors */
    s->e0[X] = 1; s->e0[Y] = 0; s->e0[Z] = 0;
    s->e1[X] = 0; s->e1[Y] = 1; s->e1[Z] = 0;
    s->e2[X] = 0; s->e2[Y] = 0; s->e2[Z] = 1;

    /* ini inertia tensor */
    float I[6]; 
#ifdef spdir // open geometry, use particles
    init_I_frompp(pp, n, pmass, com, /**/ I);
    s->mass = n*pmass;
#else
    init_I_fromm(pmass, nt, tt, vv, /**/ I);
    s->mass = mesh::volume(nt, tt, vv) * numberdensity * pmass;
#endif
        
    linal::inv3x3(I, /**/ s->Iinv);
        
    /* initial positions */
    for (int ip = 0; ip < n; ++ip) {
        float *ro = &rr0[3*ip];
        const float *r0 = pp[ip].r;
        ro[X] = r0[X]-com[X]; ro[Y] = r0[Y]-com[Y]; ro[Z] = r0[Z]-com[Z];
    }
}

void mesh2pp_hst(const Solid *ss_hst, const int ns, int nv, const float *vv, /**/ Particle *pp) {
    for (int j = 0; j < ns; ++j) {
        const Solid *s = ss_hst + j;
        update_r_hst(vv, nv, s->com, s->e0, s->e1, s->e2, /**/ pp + j * nv);

        for (int i = 0; i < nv; ++i) {
            float *v = pp[j*nv + i].v;
            v[X] = v[Y] = v[Z] = 0;
        }
    }
}
