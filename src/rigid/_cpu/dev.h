namespace rig {
namespace dev {

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

static void update_r(const float *rr0, const int n, const float *com, const float *e0, const float *e1, const float *e2, /**/ Particle *pp) {
    for (int ip = 0; ip < n; ++ip) {
        float *r0 = pp[ip].r;
        const float* ro = &rr0[3*ip];
        float x = ro[X], y = ro[Y], z = ro[Z];
        r0[X] = x*e0[X] + y*e1[X] + z*e2[X];
        r0[Y] = x*e0[Y] + y*e1[Y] + z*e2[Y];
        r0[Z] = x*e0[Z] + y*e1[Z] + z*e2[Z];

        r0[X] += com[X]; r0[Y] += com[Y]; r0[Z] += com[Z];
    }
}

void update_1s(const Force *ff, const float *rr0, int n, /**/ Particle *pp, Solid *s) {
    /* clear velocity */
    for (int ip = 0; ip < n; ++ip) {
        float *v0 = pp[ip].v;
        v0[X] = v0[Y] = v0[Z] = 0;
    }

    add_f(ff, n, /**/ s->fo);
    add_to(pp, ff, n, s->com, /**/ s->to);

    update_v (s->mass, s->fo, /**/ s->v);
    update_om(s->Iinv, s->to, /**/ s->om);

    if (pin_axis) constrain_om(/**/ s->om);
    
    if (!pin_com) add_v(s->v, n, /**/ pp);
    add_om(s->com, s->om, n, /**/ pp);

    if (pin_com) s->v[X] = s->v[Y] = s->v[Z] = 0;

    if (!pin_com) update_com(s->v, /**/ s->com);

    rot_e(s->om, /**/ s->e0);
    rot_e(s->om, /**/ s->e1);
    rot_e(s->om, /**/ s->e2);
    gram_schmidt(/**/ s->e0, s->e1, s->e2);

    update_r(rr0, n, s->com, s->e0, s->e1, s->e2, /**/ pp);
}

} // dev
} // rig
