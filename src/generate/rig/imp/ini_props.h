#ifdef spdir // open geometry, use particles    
static void init_I_frompp(const Particle *pp, int n, float pmass, const float *com, /**/ float *I) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    enum {YX = XY, ZX = XZ, ZY = YZ};

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
    mesh_center_of_mass(nt, tt, vv, /**/ com);
    mesh_inertia_tensor(nt, tt, vv, com, numberdensity, /**/ I);

    for (int c = 0; c < 6; ++c) I[c] *= pmass;
}
#endif    

void ini_props(int n, const Particle *pp, float pmass, const float *com, int nt, const int4 *tt, const float *vv, /**/ float *rr0, Solid *s) {
    enum {X, Y, Z};
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
    s->mass = n * pmass;
#else
    init_I_fromm(pmass, nt, tt, vv, /**/ I);
    s->mass = mesh_volume(nt, tt, vv) * numberdensity * pmass;
#endif
    UC(linal_inv3x3(I, /**/ s->Iinv));
    /* initial positions */
    for (int ip = 0; ip < n; ++ip) {
        float *ro = &rr0[3*ip];
        const float *r0 = pp[ip].r;
        ro[X] = r0[X]-com[X]; ro[Y] = r0[Y]-com[Y]; ro[Z] = r0[Z]-com[Z];
    }
}
