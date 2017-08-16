namespace mbounce {
namespace sub {

enum {X, Y, Z};

#ifdef FORWARD_EULER
_DH_ void rvprev(const float *r1, const float *v1, const float *f0, /**/ float *r0, float *v0) {
    for (int c = 0; c < 3; ++c) {
        v0[c] = v1[c] - f0[c] * dt;
        r0[c] = r1[c] - v0[c] * dt;
    }
}
#else // velocity-verlet
_DH_ void rvprev(const float *r1, const float *v1, const float *, /**/ float *r0, float *v0) {
     for (int c = 0; c < 3; ++c) {
        r0[c] = r1[c] - v1[c] * dt;
        //v0[c] = v1[c] - f0[c] * dt;

        // BB assumes r0 + v0 dt = r1 for now
        v0[c] = v1[c];
    }
}
#endif

_DH_ void bounce_back(const Particle *p0, const float *rw, const float *vw, const float h, /**/ Particle *pn) {
    pn->v[X] = 2 * vw[X] - p0->v[X];
    pn->v[Y] = 2 * vw[Y] - p0->v[Y];
    pn->v[Z] = 2 * vw[Z] - p0->v[Z];

    pn->r[X] = rw[X] + (dt-h) * pn->v[X];
    pn->r[Y] = rw[Y] + (dt-h) * pn->v[Y];
    pn->r[Z] = rw[Z] + (dt-h) * pn->v[Z];
}

_DH_ void lin_mom_change(const float v0[3], const float v1[3], /**/ float dP[3]) {
    dP[X] = -(v1[X] - v0[X]);
    dP[Y] = -(v1[Y] - v0[Y]);
    dP[Z] = -(v1[Z] - v0[Z]);
}

_DH_ void ang_mom_change(const float r[3], const float v0[3], const float v1[3], /**/ float dL[3]) {
    dL[X] = -(r[Y] * v1[Z] - r[Z] * v1[Y]  -  r[Y] * v0[Z] + r[Z] - v0[Y]);
    dL[X] = -(r[Z] * v1[X] - r[X] * v1[Z]  -  r[Z] * v0[X] + r[X] - v0[Z]);
    dL[X] = -(r[X] * v1[Y] - r[Y] * v1[X]  -  r[X] * v0[Y] + r[Y] - v0[X]);
}

/* shift origin from 0 to R for ang momentum */
_DH_ void mom_shift_ref(const float R[3], /**/ Momentum *m) {
    m->L[0] -= R[1] * m->P[2] - R[2] * m->P[1];
    m->L[1] -= R[2] * m->P[0] - R[0] * m->P[2];
    m->L[2] -= R[0] * m->P[1] - R[1] * m->P[0];
}

static _DH_ bool nz(float a) {return fabs(a) > 1e-6f;}
_DH_ bool nonzero(const Momentum *m) {
    return nz(m->P[0]) && nz(m->P[1]) && nz(m->P[2]) &&
        nz(m->L[0]) && nz(m->L[1]) && nz(m->L[2]);
}

} // sub
} // mbounce
