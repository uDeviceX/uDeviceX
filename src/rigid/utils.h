namespace rig {
namespace dev {

static _HD_ float dot(const float *v, const float *u) {
    return v[X]*u[X] + v[Y]*u[Y] + v[Z]*u[Z];
}

static _HD_ void reject(/**/ float *v, float *u) {
    const float d = dot(v, u);
    v[X] -= d*u[X]; v[Y] -= d*u[Y]; v[Z] -= d*u[Z];
}

static _HD_ float norm(const float *v) {
    return sqrt(v[X]*v[X] + v[Y]*v[Y] + v[Z]*v[Z]);
}

static _HD_ void normalize(/**/ float *v) {
    const float s = 1.f / norm(v);
    v[X] *= s; v[Y] *= s; v[Z] *= s;
}

static _HD_ void gram_schmidt(/**/ float *e0, float *e1, float *e2) {
    /* :TODO: use better more stable version of Gram-Schmidt */
    normalize(e0);

    reject(e1, e0);
    normalize(e1);

    reject(e2, e0);
    reject(e2, e1);
    normalize(e2);
}

static _HD_ void rot_e(const float *om, /**/ float *e) {
    const float omx = om[X], omy = om[Y], omz = om[Z];
    const float ex = e[X], ey = e[Y], ez = e[Z];

    const float vx = omy*ez - omz*ey;
    const float vy = omz*ex - omx*ez;
    const float vz = omx*ey - omy*ex;

    e[X] += vx*dt; e[Y] += vy*dt; e[Z] += vz*dt;
}

} // dev
} // rig
