
namespace rig {

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
static void init_I_fromm(float pmass, const Mesh mesh, /**/ float *I) {
    float com[3] = {0};
    mesh::center_of_mass(mesh, /**/ com);
    mesh::inertia_tensor(mesh, com, numberdensity, /**/ I);

    for (int c = 0; c < 6; ++c) I[c] *= pmass;
}
#endif
    
void ini(const Particle *pp, int n, float pmass, const float *com, const Mesh mesh, /**/ float *rr0, Solid *s) {
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
    init_I_fromm(pmass, mesh, /**/ I);
    s->mass = mesh::volume(mesh) * numberdensity * pmass;
#endif
        
    l::linal::inv3x3(I, /**/ s->Iinv);
        
    /* initial positions */
    for (int ip = 0; ip < n; ++ip) {
        float *ro = &rr0[3*ip];
        const float *r0 = pp[ip].r;
        ro[X] = r0[X]-com[X]; ro[Y] = r0[Y]-com[Y]; ro[Z] = r0[Z]-com[Z];
    }
}

void dump(const int it, const Solid *ss, const Solid *ssbb, int ns, const int *mcoords) {
    static bool first = true;
    char fname[256];

    for (int j = 0; j < ns; ++j) {
        const Solid *s   = ss   + j;
        const Solid *sbb = ssbb + j;
            
        sprintf(fname, DUMP_BASE "/solid_diag_%04d.txt", (int) s->id);
        FILE *fp;
        if (first) fp = fopen(fname, "w");
        else       fp = fopen(fname, "a");

        fprintf(fp, "%+.6e ", dt*it);

        auto write_v = [fp] (const float *v) {
            fprintf(fp, "%+.6e %+.6e %+.6e ", v[X], v[Y], v[Z]);
        };

        // make global coordinates
        float com[3];
        {
            const int L[3] = {XS, YS, ZS};
            int mi[3];
            for (int c = 0; c < 3; ++c) mi[c] = (mcoords[c] + 0.5) * L[c];
            for (int c = 0; c < 3; ++c) com[c] = s->com[c] + mi[c];
        }
            
        write_v(com);
        write_v(s->v );
        write_v(s->om);
        write_v(s->fo);
        write_v(s->to);
        write_v(s->e0);
        write_v(s->e1);
        write_v(s->e2);
        write_v(sbb->fo);
        write_v(sbb->to);
        fprintf(fp, "\n");
        
        fclose(fp);
    }

    first = false;
}

} // rig
