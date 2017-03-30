#include "k/solid.h"

namespace solid {

#define X 0
#define Y 1
#define Z 2
#define XX 0
#define XY 1
#define XZ 2
#define YY 3
#define YZ 4
#define ZZ 5

#define YX XY
#define ZX XZ
#define ZY YZ

    
    bool inside(float x, float y, float z) {
        return k_solid::shape::inside(x, y, z);
    }

    void init_com(Particle *pp, int n, /**/ float *com) {
        com[X] = com[Y] = com[Z] = 0;
        for (int ip = 0; ip < n; ++ip) {
            float *r0 = pp[ip].r;
            com[X] += r0[X]; com[Y] += r0[Y]; com[Z] += r0[Z];
        }
        com[X] /= n; com[Y] /= n; com[Z] /= n;
        k_solid::pbc_solid(com);
    }

    void init_I(Particle *pp, int n, float mass, float *com, /**/ float *I) {
        int c;

        for (int c = 0; c < 6; ++c) I[c] = 0;

        for (int ip = 0; ip < n; ++ip) {
            float *r0 = pp[ip].r;
            float x = r0[X]-com[X], y = r0[Y]-com[Y], z = r0[Z]-com[Z];
            I[XX] += y*y + z*z;
            I[YY] += z*z + x*x;
            I[ZZ] += x*x + y*y;
            I[XY] -= x*y;
            I[XZ] -= z*x;
            I[YZ] -= y*z;
        }

        for (c = 0; c < 6; ++c) I[c] *= mass;
    }

    void init(Particle *pp, int n, float mass,
              /**/ float *rr0, float *Iinv, float *com, float *e0, float *e1, float *e2, float *v, float *om) {
        v[X] = v[Y] = v[Z] = 0; 
        om[X] = om[Y] = om[Z] = 0; 

        /* init basis vectors */
        e0[X] = 1; e0[Y] = 0; e0[Z] = 0;
        e1[X] = 0; e1[Y] = 1; e1[Z] = 0;
        e2[X] = 0; e2[Y] = 0; e2[Z] = 1;

        if (!pin_com) init_com(pp, n, /**/ com);
        else com[X] = com[Y] = com[Z] = 0;

        /* init inertia tensor */
        float I[6]; solid::init_I(pp, n, mass, com, /**/ I);
        gsl::inv3x3(I, /**/ Iinv);

        /* initial positions */
        for (int ip = 0; ip < n; ++ip) {
            float *ro = &rr0[3*ip];
            float *r0 = pp[ip].r;
            ro[X] = r0[X]-com[X]; ro[Y] = r0[Y]-com[Y]; ro[Z] = r0[Z]-com[Z];
        }
    }
    
    void add_f(Force *ff, int n, /**/ float *f) {
        for (int ip = 0; ip < n; ++ip) {
            float *f0 = ff[ip].f;
            f[X] += f0[X]; f[Y] += f0[Y]; f[Z] += f0[Z];
        }
    }

    void add_to(Particle *pp, Force *ff, int n, float *com, /**/ float *to) {
        for (int ip = 0; ip < n; ++ip) {
            float *r0 = pp[ip].r, *f0 = ff[ip].f;
            float x = r0[X]-com[X], y = r0[Y]-com[Y], z = r0[Z]-com[Z];
            float fx = f0[X], fy = f0[Y], fz = f0[Z];
            to[X] += y*fz - z*fy;
            to[Y] += z*fx - x*fz;
            to[Z] += x*fy - y*fx;
        }
    }

    void update_om(float *Iinv, float *to, /**/ float *om) {
        float *A = Iinv, *b = to, dom[3];
        dom[X] = A[XX]*b[X] + A[XY]*b[Y] + A[XZ]*b[Z];
        dom[Y] = A[YX]*b[X] + A[YY]*b[Y] + A[YZ]*b[Z];
        dom[Z] = A[ZX]*b[X] + A[ZY]*b[Y] + A[ZZ]*b[Z];

        om[X] += dom[X]*dt; om[Y] += dom[Y]*dt; om[Z] += dom[Z]*dt;
    }

    void update_v(float mass, float *f, int n, /**/ float *v) {
        float sc = dt/(mass*n);
        v[X] += f[X]*sc; v[Y] += f[Y]*sc; v[Z] += f[Z]*sc;
    }

    void add_v(float *v, int n, /**/ Particle *pp) {
        for (int ip = 0; ip < n; ++ip) {
            float *v0 = pp[ip].v;
            lastbit::Preserver up(v0[X]);
            v0[X] += v[X]; v0[Y] += v[Y]; v0[Z] += v[Z];
        }
    }

    void add_om(float *com, float *om, int n, /**/ Particle *pp) {
        float omx = om[X], omy = om[Y], omz = om[Z];
        for (int ip = 0; ip < n; ++ip) {
            float *r0 = pp[ip].r, *v0 = pp[ip].v;
            float x = r0[X]-com[X], y = r0[Y]-com[Y], z = r0[Z]-com[Z];
            lastbit::Preserver up(v0[X]);
            v0[X] += omy*z - omz*y;
            v0[Y] += omz*x - omx*z;
            v0[Z] += omx*y - omy*x;
        }
    }

    void constrain_om(/**/ float *om) {
        om[X] = om[Y] = 0;
    }

    void update_com(float *v, /**/ float *com) {
        com[X] += v[X]*dt; com[Y] += v[Y]*dt; com[Z] += v[Z]*dt;
        k_solid::pbc_solid(/**/ com);
    }

    void update_r(float *rr0, int n, float *com, float *e0, float *e1, float *e2, /**/ Particle *pp) {
        for (int ip = 0; ip < n; ++ip) {
            float *r0 = pp[ip].r, *ro = &rr0[3*ip];
            float x = ro[X], y = ro[Y], z = ro[Z];
            r0[X] = x*e0[X] + y*e1[X] + z*e2[X];
            r0[Y] = x*e0[Y] + y*e1[Y] + z*e2[Y];
            r0[Z] = x*e0[Z] + y*e1[Z] + z*e2[Z];

            r0[X] += com[X]; r0[Y] += com[Y]; r0[Z] += com[Z];
        }
    }

    void reinit_f_to(/**/ float *f, float *to)
    {
        f[X] = f[Y] = f[Z] = 0;
        to[X] = to[Y] = to[Z] = 0;
    }

    void update(Force *ff, float *rr0, int n, /**/ Particle *pp, Solid *shst)
    {
        /* clear velocity */
        for (int ip = 0; ip < n; ++ip) {
            float *v0 = pp[ip].v;
            lastbit::Preserver up(v0[X]);
            v0[X] = v0[Y] = v0[Z] = 0;
        }

        add_f(ff, n, /**/ shst->fo);
        add_to(pp, ff, n, shst->com, /**/ shst->to);

        update_v(rbc_mass, shst->fo, n, /**/ shst->v);
        update_om(shst->Iinv, shst->to, /**/ shst->om);

        if (pin_axis) constrain_om(/**/ shst->om);
    
        if (!pin_com) add_v(shst->v, n, /**/ pp);
        add_om(shst->com, shst->om, n, /**/ pp);

        if (pin_com) shst->v[X] = shst->v[Y] = shst->v[Z] = 0;

        if (!pin_com) update_com(shst->v, /**/ shst->com);

        k_solid::rot_e(shst->om, /**/ shst->e0);
        k_solid::rot_e(shst->om, /**/ shst->e1);
        k_solid::rot_e(shst->om, /**/ shst->e2);
        k_solid::gram_schmidt(/**/ shst->e0, shst->e1, shst->e2);

        update_r(rr0, n, shst->com, shst->e0, shst->e1, shst->e2, /**/ pp);
    }

    void update_nohost(const Force *ff, const float *rr0, const int n, /**/ Particle *pp, Solid *sdev)
    {
        k_solid::add_f_to <<<k_cnf(n)>>> (pp, ff, n, sdev->com, /**/ sdev->fo, sdev->to);

        //TODO mass
        k_solid::update_om_v <<<1, 1>>> (rbc_mass, sdev->Iinv, sdev->fo, sdev->to, /**/ sdev->om, sdev->v);

        k_solid::compute_velocity <<<k_cnf(n)>>> (sdev->v, sdev->com, sdev->om, n, /**/ pp);
        
        if (!pin_com) k_solid::update_com <<<1, 1>>> (sdev->v, /**/ sdev->com);

        k_solid::rot_referential <<<1, 1>>> (sdev->om, /**/ sdev->e0, sdev->e1, sdev->e2);

        k_solid::update_r <<<k_cnf(n)>>> (rr0, n, sdev->com, sdev->e0, sdev->e1, sdev->e2, /**/ pp);
    }

    void dump(const int it, const float *com, const float *v, const float *om, const float *to) {
        static bool first = true;
        const char *fname = "solid_diag.txt";
        FILE *fp;
        if (first) fp = fopen(fname, "w");
        else       fp = fopen(fname, "a");
        first = false;

        fprintf(fp, "%+.6e ", dt*it);
        fprintf(fp, "%+.6e %+.6e %+.6e ", com[X], com[Y], com[Z]);
        fprintf(fp, "%+.6e %+.6e %+.6e ",   v[X],   v[Y],   v[Z]);
        fprintf(fp, "%+.6e %+.6e %+.6e ",  om[X],  om[Y],  om[Z]);
        fprintf(fp, "%+.6e %+.6e %+.6e\n", to[X],  to[Y],  to[Z]);

        fclose(fp);
    }

#undef X
#undef Y
#undef Z
#undef XX
#undef XY
#undef XZ
#undef YY
#undef YZ
#undef ZZ

#undef YX
#undef ZX
#undef ZY
}
