#include <stdlib.h>
#include <math.h>

#include "type.h"
#include "utils.h"
#include "particles.h"

static const real yang_coeff_3d = 2.0 / M_PI;

static void clear_vel(int n, Particles *p) {
    for (int i = 0; i < n; ++i)
        p->vx[i] = p->vy[i] = p->vz[i] = 0;
}

void particles_clear_forces(int n, Particles *p) {
    for (int i = 0; i < n; ++i)
        p->fx[i] = p->fy[i] = p->fz[i] = 0;
}

static real gen(int L) {
    return L * (drand48() - 0.5);
}

static void ini_array(int n, real **a) {
    *a = (real*) malloc(n*sizeof(real));
}

void particles_ini(int n, int Lx, int Ly, int Lz, Particles *p) {
    int i;
    ini_array(n, &p->xx); ini_array(n, &p->yy); ini_array(n, &p->zz);
    ini_array(n, &p->vx); ini_array(n, &p->vy); ini_array(n, &p->vz);
    ini_array(n, &p->fx); ini_array(n, &p->fy); ini_array(n, &p->fz);
    clear_vel(n, p);
    particles_clear_forces(n, p);

    for (i = 0; i < n; ++i) {
        p->xx[i] = gen(Lx);
        p->yy[i] = gen(Ly);
        p->zz[i] = gen(Lz);
    }
}

void particles_fin(Particles *p) {
    free(p->xx); free(p->yy); free(p->zz);
    free(p->vx); free(p->vy); free(p->vz);
    free(p->fx); free(p->fy); free(p->fz);
}

static real periodic_advance(real x, real dx, int L) {
    x += dx;
    if (x < - 0.5 * L) x += L;
    if (x >=  0.5 * L) x -= L;
    return x;
}

void particles_advance(real dt, int Lx, int Ly, int Lz, int n, Particles *p) {
    for (int i = 0; i < n; ++i) {
        p->vx[i] += dt * p->fx[i];
        p->vy[i] += dt * p->fy[i];
        p->vz[i] += dt * p->fz[i];
        
        p->xx[i] = periodic_advance(p->xx[i], dt * p->vx[i], Lx);
        p->yy[i] = periodic_advance(p->yy[i], dt * p->vy[i], Ly);
        p->zz[i] = periodic_advance(p->zz[i], dt * p->vz[i], Lz);
    }
}

static void fetch(int i, const Particles *p, real r[]) {
    enum {X, Y, Z};
    r[X] = p->xx[i];
    r[Y] = p->yy[i];
    r[Z] = p->zz[i];
}

static real periodic_d(int L, const real xi, const real xj) {
    real dx0, dx;
    dx = dx0 = xi - xj;
    if (fabs(dx0 - L) < fabs(dx)) dx = dx0 - L;
    if (fabs(dx0 + L) < fabs(dx)) dx = dx0 + L;
    return dx;
}

static void periodic_dr(int Lx, int Ly, int Lz, const real ri[], const real rj[], real dr[]) {
    enum {X, Y, Z};
    dr[X] = periodic_d(Lx, ri[X], rj[X]);
    dr[Y] = periodic_d(Ly, ri[Y], rj[Y]);
    dr[Z] = periodic_d(Lz, ri[Z], rj[Z]);
}

static real yang_w0(real q) {
    if      (q < 1) return cu(2 - q) - 4 * cu(1 - q);
    else if (q < 2) return cu(2 - q);
    else            return 0;
}

static real yang_w3(real rc, real r) {
    real q;
    q = 2 * r / rc;
    return yang_coeff_3d * yang_w0(q) / (rc * rc * rc);
}

static real kernel(real rc, real r) {
    if (r < rc) return 1.0 - r / rc;
    return 0;
    // return yang_w3(rc, r);
}

static real force(real rc, const real dr[], real f[]) {
    enum {X, Y, Z, D};
    f[X] = f[Y] = f[Z] = 0;
    real f0, rsq, r, inv_r;

    rsq = sq(dr[X]) + sq(dr[Y]) + sq(dr[Z]);
    r = sqrt(rsq);
    if (r > rc) return r;
    
    inv_r = 1.0 / r;
    
    // f0 = yang_w3(rc, r);
    f0 = kernel(rc, r);
    f0 *= inv_r;

    f[X] = dr[X] * f0;
    f[Y] = dr[Y] * f0;
    f[Z] = dr[Z] * f0;
    return r;
}

real particles_interactions(real rc, int Lx, int Ly, int Lz, int n, Particles *p) {
    enum {X, Y, Z, D};
    int i, j;
    real ri[D], rj[D], dr[D], f[D], fi[D], r, rmin;
    rmin = sqrt(sq(Lx) + sq(Ly) + sq(Lz));

    // printf("%d\n", n);
    
    for (i = 0; i < n; ++i) {
        fetch(i, p, ri);
        fi[X] = fi[Y] = fi[Z] = 0;
        
        for (j = 0; j < i; ++j) {
            fetch(j, p, rj);
            periodic_dr(Lx, Ly, Lz, ri, rj, dr);            
            r = force(rc, dr, f);

            fi[X] += f[X];
            fi[Y] += f[Y];
            fi[Z] += f[Z];
            p->fx[j] -= f[X];
            p->fy[j] -= f[Y];
            p->fz[j] -= f[Z];

            rmin = r < rmin ? r : rmin;
        }
        p->fx[i] += fi[X];
        p->fy[i] += fi[Y];
        p->fz[i] += fi[Z];
    }
    return rmin;
}

real particles_temperature(int n, const Particles *p) {
    real vsq, sum_vsq = 0;
    int i;
    for (i = 0; i < n; ++i) {
        vsq = sq(p->vx[i]) + sq(p->vy[i]) + sq(p->vz[i]);
        sum_vsq += vsq;
    }
    return sum_vsq / (3*n);
}

void particles_rescale_v(real T0, real T, int n, Particles *p) {
    real s;
    int i;
    s = sqrt(T0 / T);
    for (i = 0; i < n; ++i) {
        p->vx[i] *= s;
        p->vy[i] *= s;
        p->vz[i] *= s;
    }
}

void particles_shift(real x, real y, real z, int n, Particles *p) {
    int i;
    for (i = 0; i < n; ++i) {
        p->xx[i] += x;
        p->yy[i] += y;
        p->zz[i] += z;
    }
}
