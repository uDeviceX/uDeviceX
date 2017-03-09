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

void reject(float *v, float *u) {
    float d = dot(v, u);
    v[X] -= d*u[X]; v[Y] -= d*u[Y]; v[Z] -= d*u[Z];
}

float norm(float *v) {
    return sqrt(v[X]*v[X]+v[Y]*v[Y]+v[Z]*v[Z]);
}

void normalize(float *v) {
    float nrm = norm(v);
    v[X] /= nrm; v[Y] /= nrm; v[Z] /= nrm;
}

void gram_schmidt(float *e0, float *e1, float *e2) {
    normalize(e0);

    reject(e1, e0);
    normalize(e1);

    reject(e2, e0);
    reject(e2, e1);
    normalize(e2);
}

void rotate_e(float *e, float *om) {
    float omx = om[X], omy = om[Y], omz = om[Z];
    float ex = e[X], ey = e[Y], ez = e[Z];
    float vx, vy, vz;
    vx = omy*ez - omz*ey;
    vy = omz*ex - omx*ez;
    vz = omx*ey - omy*ex;
    e[X] += vx*dt; e[Y] += vy*dt; e[Z] += vz*dt;
}

/* wrap COM to the domain; TODO: many processes */
void pbc_solid(float *com) {
    float lo[3] = {-0.5*XS, -0.5*YS, -0.5*ZS};
    float hi[3] = { 0.5*XS,  0.5*YS,  0.5*ZS};
    float L[3] = {XS, YS, ZS};
    for (int c = 0; c < 3; ++c) {
        while (com[c] <  lo[c]) com[c] += L[c];
        while (com[c] >= hi[c]) com[c] -= L[c];
    }
}

void compute_torque(Particle *pp, Force *ff, int n, float *com, /**/ float *to) {
    to[X] = to[Y] = to[Z] = 0;
    for (int ip = 0; ip < n; ++ip) {
        float *r0 = pp[ip].r, *f0 = ff[ip].f;
        float x = r0[X]-com[X], y = r0[Y]-com[Y], z = r0[Z]-com[Z];
        float fx = f0[X], fy = f0[Y], fz = f0[Z];
        to[X] += y*fz - z*fy;
        to[Y] += z*fx - x*fz;
        to[Z] += x*fy - y*fx;
    }
}

void update_omega(float *Iinv, float *to, /**/ float *om) {
    float *A = Iinv, *b = to, dom[3];
    dom[X] = A[XX]*b[X] + A[XY]*b[Y] + A[XZ]*b[Z];
    dom[Y] = A[YX]*b[X] + A[YY]*b[Y] + A[YZ]*b[Z];
    dom[Z] = A[ZX]*b[X] + A[ZY]*b[Y] + A[ZZ]*b[Z];

    om[X] += dom[X]*dt; om[Y] += dom[Y]*dt; om[Z] += dom[Z]*dt;
}
}
