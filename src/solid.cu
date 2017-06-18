#include <gsl/gsl_linalg.h>
#include "gsl.impl.h"

#include <mpi.h>
#include "common.h"
#include <conf.h>
#include "conf.common.h"

#include "solid.h"
#include "k/solid.h"
#include "mesh.h"

namespace solid
{
enum {X, Y, Z};
enum {XX, XY, XZ, YY, YZ, ZZ};
enum {YX = XY, ZX = XZ, ZY = YZ};

#ifdef spdir // open geometry, use particles    
static void init_I_frompp(const Particle *pp, int n, float pmass, const float *com, /**/ float *I)
{
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
static void init_I_fromm(float pmass, const Mesh mesh, /**/ float *I)
{
    float com[3] = {0};
    mesh::center_of_mass(mesh, /**/ com);
    mesh::inertia_tensor(mesh, com, numberdensity, /**/ I);

    for (int c = 0; c < 6; ++c) I[c] *= pmass;
}
#endif
    
void ini(const Particle *pp, int n, float pmass, const float *com, const Mesh mesh, /**/ float *rr0, Solid *s)
{
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
        
    gsl::inv3x3(I, /**/ s->Iinv);

    // {
    //     FILE *f = fopen("solid_Iinv.txt", "w");

    //     fprintf(f, "%+.6e %+.6e %+.6e\n", s->Iinv[XX], s->Iinv[XY], s->Iinv[XZ]);
    //     fprintf(f, "%+.6e %+.6e %+.6e\n", s->Iinv[YX], s->Iinv[YY], s->Iinv[YZ]);
    //     fprintf(f, "%+.6e %+.6e %+.6e\n", s->Iinv[ZX], s->Iinv[ZY], s->Iinv[ZZ]);
            
    //     fclose(f);
    // }
        
    /* initial positions */
    for (int ip = 0; ip < n; ++ip) {
        float *ro = &rr0[3*ip];
        const float *r0 = pp[ip].r;
        ro[X] = r0[X]-com[X]; ro[Y] = r0[Y]-com[Y]; ro[Z] = r0[Z]-com[Z];
    }
}

static void add_f(const Force *ff, int n, /**/ float *f) {
    for (int ip = 0; ip < n; ++ip) {
        const float *f0 = ff[ip].f;
        f[X] += f0[X]; f[Y] += f0[Y]; f[Z] += f0[Z];
    }
}

static void add_to(const Particle *pp, const Force *ff, int n, const float *com, /**/ float *to) {
    for (int ip = 0; ip < n; ++ip) {
        const float *r0 = pp[ip].r, *f0 = ff[ip].f;
        const float x = r0[X]-com[X], y = r0[Y]-com[Y], z = r0[Z]-com[Z];
        const float fx = f0[X], fy = f0[Y], fz = f0[Z];
        to[X] += y*fz - z*fy;
        to[Y] += z*fx - x*fz;
        to[Z] += x*fy - y*fx;
    }
}

static void update_om(const float *Iinv, const float *to, /**/ float *om) {
    const float *A = Iinv, *b = to;
    float dom[3];
    dom[X] = A[XX]*b[X] + A[XY]*b[Y] + A[XZ]*b[Z];
    dom[Y] = A[YX]*b[X] + A[YY]*b[Y] + A[YZ]*b[Z];
    dom[Z] = A[ZX]*b[X] + A[ZY]*b[Y] + A[ZZ]*b[Z];

    om[X] += dom[X]*dt; om[Y] += dom[Y]*dt; om[Z] += dom[Z]*dt;
}

static void update_v(float mass, const float *f, /**/ float *v) {
    float sc = dt/mass;
    v[X] += f[X]*sc; v[Y] += f[Y]*sc; v[Z] += f[Z]*sc;
}

static void add_v(const float *v, int n, /**/ Particle *pp) {
    for (int ip = 0; ip < n; ++ip) {
        float *v0 = pp[ip].v;
        v0[X] += v[X]; v0[Y] += v[Y]; v0[Z] += v[Z];
    }
}

static void add_om(float *com, float *om, int n, /**/ Particle *pp) {
    float omx = om[X], omy = om[Y], omz = om[Z];
    for (int ip = 0; ip < n; ++ip) {
        float *r0 = pp[ip].r, *v0 = pp[ip].v;
        float x = r0[X]-com[X], y = r0[Y]-com[Y], z = r0[Z]-com[Z];
        v0[X] += omy*z - omz*y;
        v0[Y] += omz*x - omx*z;
        v0[Z] += omx*y - omy*x;
    }
}

static void constrain_om(/**/ float *om) {
    om[X] = om[Y] = 0;
}

static void update_com(const float *v, /**/ float *com) {
    com[X] += v[X]*dt; com[Y] += v[Y]*dt; com[Z] += v[Z]*dt;
}

static void update_r(const float *rr0, const int n, const float *com, const float *e0, const float *e1, const float *e2, /**/ Particle *pp)
{
    for (int ip = 0; ip < n; ++ip)
    {
        float *r0 = pp[ip].r;
        const float* ro = &rr0[3*ip];
        float x = ro[X], y = ro[Y], z = ro[Z];
        r0[X] = x*e0[X] + y*e1[X] + z*e2[X];
        r0[Y] = x*e0[Y] + y*e1[Y] + z*e2[Y];
        r0[Z] = x*e0[Z] + y*e1[Z] + z*e2[Z];

        r0[X] += com[X]; r0[Y] += com[Y]; r0[Z] += com[Z];
    }
}

void reinit_ft_hst(const int nsolid, /**/ Solid *ss)
{
    for (int i = 0; i < nsolid; ++i)
    {
        Solid *s = ss + i;
            
        s->fo[X] = s->fo[Y] = s->fo[Z] = 0;
        s->to[X] = s->to[Y] = s->to[Z] = 0;
    }
}

void reinit_ft_dev(const int nsolid, /**/ Solid *ss)
{
    k_solid::reinit_ft <<< k_cnf(nsolid) >>> (nsolid, /**/ ss);
}

static void update_hst_1s(const Force *ff, const float *rr0, int n, /**/ Particle *pp, Solid *shst)
{
    /* clear velocity */
    for (int ip = 0; ip < n; ++ip) {
        float *v0 = pp[ip].v;
        v0[X] = v0[Y] = v0[Z] = 0;
    }

    add_f(ff, n, /**/ shst->fo);
    add_to(pp, ff, n, shst->com, /**/ shst->to);

    update_v (shst->mass, shst->fo, /**/ shst->v);
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

void update_hst(const Force *ff, const float *rr0, int n, int nsolid, /**/ Particle *pp, Solid *shst)
{
    int start = 0;
    const int nps = n / nsolid; /* number of particles per solid */
        
    for (int i = 0; i < nsolid; ++i)
    {
        update_hst_1s(ff + start, rr0, nps, /**/ pp + start, shst + i);
        start += nps;
    }
}
    
void update_dev(const Force *ff, const float *rr0, int n, int ns, /**/ Particle *pp, Solid *ss)
{
    if (ns < 1) return;
        
    const int nps = n / ns; /* number of particles per solid */

    const dim3 nblck ( (127 + nps) / 128, ns );
    const dim3 nthrd ( 128, 1 );
        
    k_solid::add_f_to <<< nblck, nthrd >>> (pp, ff, nps, ns, /**/ ss);

    k_solid::update_om_v <<<1, ns>>> (ns, /**/ ss);

    k_solid::compute_velocity <<< nblck, nthrd >>> (ss, ns, nps, /**/ pp);

    if (!pin_com) k_solid::update_com <<<1, 3*ns >>> (ns, /**/ ss);
        
    k_solid::rot_referential <<<1, ns>>> (ns, /**/ ss);

    k_solid::update_r <<< nblck, nthrd >>> (rr0, nps, ss, ns, /**/ pp);
}

void generate_hst(const Solid *ss_hst, const int ns, const float *rr0, const int nps, /**/ Particle *pp)
{
    int start = 0;
    for (int j = 0; j < ns; ++j)
    {
        update_r(rr0, nps, ss_hst[j].com, ss_hst[j].e0, ss_hst[j].e1, ss_hst[j].e2, /**/ pp + start);
        start += nps;
    }
}

void generate_dev(const Solid *ss_dev, const int ns, const float *rr0, const int nps, /**/ Particle *pp)
{
    if (ns < 1) return;
        
    const dim3 nblck ( (127 + nps) / 128, ns );
    const dim3 nthrd ( 128, 1 );

    k_solid::update_r <<< nblck, nthrd >>> (rr0, nps, ss_dev, ns, /**/ pp);
}

void mesh2pp_hst(const Solid *ss_hst, const int ns, const Mesh m, /**/ Particle *pp)
{
    for (int j = 0; j < ns; ++j)
    {
        const Solid *s = ss_hst + j;
        update_r(m.vv, m.nv, s->com, s->e0, s->e1, s->e2, /**/ pp + j * m.nv);

        for (int i = 0; i < m.nv; ++i)
        {
            float *v = pp[j*m.nv + i].v;
            v[X] = v[Y] = v[Z] = 0;
        }
    }
}

void update_mesh_hst(const Solid *ss_hst, const int ns, const Mesh m, /**/ Particle *pp)
{
    for (int j = 0; j < ns; ++j)
    {
        const Solid *s = ss_hst + j;
                        
        for (int i = 0; i < m.nv; ++i)
        {
            const float* ro = m.vv + 3*i;
            const Particle p0 = pp[j * m.nv + i];
            float *r = pp[j * m.nv + i].r;
            float *v = pp[j * m.nv + i].v;
                
            const float x = ro[X], y = ro[Y], z = ro[Z];
            r[X] = x * s->e0[X] + y * s->e1[X] + z * s->e2[X] + s->com[X];
            r[Y] = x * s->e0[Y] + y * s->e1[Y] + z * s->e2[Y] + s->com[Y];
            r[Z] = x * s->e0[Z] + y * s->e1[Z] + z * s->e2[Z] + s->com[Z];
                
            v[X] = (r[X] - p0.r[X]) / dt;
            v[Y] = (r[Y] - p0.r[Y]) / dt;
            v[Z] = (r[Z] - p0.r[Z]) / dt;
        }
    }
}

void update_mesh_dev(const Solid *ss_dev, const int ns, const Mesh m, /**/ Particle *pp)
{
    const dim3 nthrd(128, 1);
    const dim3 nblck((m.nv + 127)/128, ns);

    k_solid::update_mesh <<< nblck, nthrd >>> (ss_dev, m.vv, m.nv, /**/ pp);
}

void dump(const int it, const Solid *ss, const Solid *ssbb, int ns, const int *mcoords)
{
    static bool first = true;
    char fname[256];

    for (int j = 0; j < ns; ++j)
    {
        const Solid *s   = ss   + j;
        const Solid *sbb = ssbb + j;
            
        sprintf(fname, "solid_diag_%04d.txt", (int) s->id);
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
}
