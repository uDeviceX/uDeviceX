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

float dot(float *v, float *u) {
    return v[X]*u[X] + v[Y]*u[Y] + v[Z]*u[Z];
}

void reject(/**/ float *v, float *u) {
    float d = dot(v, u);
    v[X] -= d*u[X]; v[Y] -= d*u[Y]; v[Z] -= d*u[Z];
}

float norm(float *v) {
    return sqrt(v[X]*v[X]+v[Y]*v[Y]+v[Z]*v[Z]);
}

void normalize(/**/ float *v) {
    float nrm = norm(v);
    v[X] /= nrm; v[Y] /= nrm; v[Z] /= nrm;
}

void gram_schmidt(/**/ float *e0, float *e1, float *e2) {
    normalize(e0);

    reject(e1, e0);
    normalize(e1);

    reject(e2, e0);
    reject(e2, e1);
    normalize(e2);
}

void rot_e(float *om, /**/ float *e) {
    float omx = om[X], omy = om[Y], omz = om[Z];
    float ex = e[X], ey = e[Y], ez = e[Z];
    float vx, vy, vz;
    vx = omy*ez - omz*ey;
    vy = omz*ex - omx*ez;
    vz = omx*ey - omy*ex;
    e[X] += vx*dt; e[Y] += vy*dt; e[Z] += vz*dt;
}

/* wrap COM to the domain; TODO: many processes */
void pbc_solid(/**/ float *com) {
    float lo[3] = {-0.5*XS, -0.5*YS, -0.5*ZS};
    float hi[3] = { 0.5*XS,  0.5*YS,  0.5*ZS};
    float L[3] = {XS, YS, ZS};
    for (int c = 0; c < 3; ++c) {
        while (com[c] <  lo[c]) com[c] += L[c];
        while (com[c] >= hi[c]) com[c] -= L[c];
    }
}

#if defined(rsph)

#define shape sphere
#define pin_axis (false)

namespace sphere
{
    bool inside(float x, float y, float z) {
        return x*x + y*y + z*z < rsph*rsph;
    }
}
#endif

#if defined(rcyl)

#define shape cylinder
#define pin_axis (true)

namespace cylinder
{
    bool inside(float x, float y, float z) {
        return x*x + y*y < rcyl * rcyl;
    }
}
#endif

bool inside(float x, float y, float z) {
    return shape::inside(x, y, z);
}

void init_com(Particle *pp, int n, /**/ float *com) {
    com[X] = com[Y] = com[Z] = 0;
    for (int ip = 0; ip < n; ++ip) {
        float *r0 = pp[ip].r;
        com[X] += r0[X]; com[Y] += r0[Y]; com[Z] += r0[Z];
    }
    com[X] /= n; com[Y] /= n; com[Z] /= n;
    pbc_solid(com);
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

    init_com(pp, n, /**/ com);

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
    pbc_solid(/**/ com);
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

void update(Force *ff, float *rr0, int n, float mass,
        Particle *pp,
        /**/ float *Iinv, float *com, float *e0, float *e1, float *e2,
             float *v, float *om, float *f, float *to) {
    /* clear velocity */
    for (int ip = 0; ip < n; ++ip) {
        float *v0 = pp[ip].v;
        lastbit::Preserver up(v0[X]);
        v0[X] = v0[Y] = v0[Z] = 0;
    }

    add_f(ff, n, /**/ f);
    add_to(pp, ff, n, com, /**/ to);

    update_v(mass, f, n, /**/ v);
    update_om(Iinv, to, /**/ om);

    if (pin_axis) constrain_om(/**/ om);
    
    if (!pin_com) add_v(v, n, /**/ pp);
    add_om(com, om, n, /**/ pp);

    if (pin_com) v[X] = v[Y] = v[Z] = 0;

    if (!pin_com) update_com(v, /**/ com);
    rot_e(om, /**/ e0); rot_e(om, /**/ e1); rot_e(om, /**/ e2); gram_schmidt(/**/ e0, e1, e2);

    update_r(rr0, n, com, e0, e1, e2, /**/ pp);
}

void dump(int it, float *com, float *v, float *om, float *to) {
    static bool first = true;
    const char *fname = "solid_diag.txt";
    FILE *fp;
    if (first) fp = fopen(fname, "w");
    else       fp = fopen(fname, "a");
    first = false;

    fprintf(fp, "%g ", dt*it);
    fprintf(fp, "%g %g %g ", com[X], com[Y], com[Z]);
    fprintf(fp, "%g %g %g ", v[X], v[Y], v[Z]);
    fprintf(fp, "%g %g %g ", om[X], om[Y], om[Z]);
    fprintf(fp, "%g %g %g\n", to[X], to[Y], to[Z]);

    fclose(fp);
}
}
