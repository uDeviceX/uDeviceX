#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <math.h>

struct Args {
    int n;
    int Lx, Ly, Lz;
};

typedef float real;

struct Particles {
    real *xx, *yy, *zz;
    real *fx, *fy, *fz;
};

static const real yang_coeff_3d = 2.0 / M_PI;

static void usg() {
    fprintf(stderr,
            "usage: obj-ic <N> <Lx> <Ly> <Lz>\n"
            "\t N          : number of spheres to pack\n"
            "\t Lx, Ly, Lz : dimension of domain\n");
    exit(1);
}

static bool shift(int *c, char ***v) {
    (*c) --; (*v)++;
    return (*c) >= 0;
}

static void parse(int c, char **v, Args *a) {
    if (!shift(&c, &v)) usg();
    a->n = atoi(*v);

    if (!shift(&c, &v)) usg();
    a->Lx = atoi(*v);
    if (!shift(&c, &v)) usg();
    a->Ly = atoi(*v);
    if (!shift(&c, &v)) usg();
    a->Lz = atoi(*v);
}

static void clear_forces(int n, Particles *p) {
    for (int i = 0; i < n; ++i)
        p->fx[i] = p->fy[i] = p->fz[i] = 0;
}

static real gen(int L) {
    return L * (drand48() - 0.5);
}

static void particles_ini(int n, int Lx, int Ly, int Lz, Particles *p) {
    size_t sz = n * sizeof(real);
    int i; 
    p->xx = (real*) malloc(sz);
    p->yy = (real*) malloc(sz);
    p->zz = (real*) malloc(sz);
    p->fx = (real*) malloc(sz);
    p->fy = (real*) malloc(sz);
    p->fz = (real*) malloc(sz);

    for (i = 0; i < n; ++i) {
        p->xx[i] = gen(Lx);
        p->yy[i] = gen(Ly);
        p->zz[i] = gen(Lz);
    }
}

static void particles_fin(Particles *p) {
    free(p->xx); free(p->yy); free(p->zz);
    free(p->fx); free(p->fy); free(p->fz);
}

static real periodic_advance(real dt, real x, real f, int L) {
    x += dt * f;
    if (x < - 0.5 * L) x += L;
    if (x >=  0.5 * L) x -= L;
    return x;
}

static void advance(real dt, int Lx, int Ly, int Lz, int n, Particles *p) {
    for (int i = 0; i < n; ++i) {
        p->xx[i] = periodic_advance(dt, p->xx[i], p->fx[i], Lx);
        p->yy[i] = periodic_advance(dt, p->yy[i], p->fy[i], Ly);
        p->zz[i] = periodic_advance(dt, p->zz[i], p->fz[i], Lz);
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
    if (fabs(dx0 - L) < dx) dx = dx0 - L;
    if (fabs(dx0 + L) < dx) dx = dx0 + L;
    return dx;
}

static void periodic_dr(int Lx, int Ly, int Lz, const real ri[], const real rj[], real dr[]) {
    enum {X, Y, Z};
    dr[X] = periodic_d(Lx, ri[X], rj[X]);
    dr[Y] = periodic_d(Ly, ri[Y], rj[Y]);
    dr[Z] = periodic_d(Lz, ri[Z], rj[Z]);
}

static real yang_w0(real q) {
    if      (q < 1) return pow(2 - q, 3) - 4 * pow(1 - q, 3);
    else if (q < 2) return pow(2 - q, 3);
    else            return 0;
}

static real yang_w3(real rc, real r) {
    real q;
    q = 2 * r / rc;
    return yang_coeff_3d * yang_w0(q) / (rc * rc * rc);
}

static real force(real rc, const real dr[], real f[]) {
    enum {X, Y, Z, D};
    f[X] = f[Y] = f[Z] = 0;
    real f0, rsq, r, inv_r;

    rsq = dr[X] * dr[X] + dr[Y] * dr[Y] + dr[Z] * dr[Z];
    r = sqrt(rsq);
    if (r > rc || r < 1e-6) return r;
    
    inv_r = 1.0 / r;
    
    f0 = yang_w3(rc, r);
    f0 *= inv_r;

    f[X] = dr[X] * f0;
    f[Y] = dr[Y] * f0;
    f[Z] = dr[Z] * f0;
    return r;
}

static bool interactions(real rc, int Lx, int Ly, int Lz, int n, Particles *p) {
    enum {X, Y, Z, D};
    int i, j;
    real ri[D], rj[D], dr[D], f[D], fi[D], r, rmin;
    rmin = sqrt(Lx * Lx + Ly * Lz + Lz * Lz);

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

    printf("rmin = %g\n", rmin);
    return rmin > rc / 2;
}

static void dump_xyz(const char *fname, int n, const Particles *p) {
    FILE *f = fopen(fname, "w");
    fprintf(f, "%d\n\n", n);
    for (int i = 0; i < n; ++i) {
        fprintf(f, "O %g %g %g\n",
                p->xx[i], p->yy[i], p->zz[i]);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    Args a;
    Particles p;
    long seed = 12345;
    bool converged = false;
    int step = 0;

    static const real dt = 0.1;
    static const real rc = 4.0;
    static const int max_steps = 10000;
    
    parse(argc, argv, &a);
    srand(seed);

    particles_ini(a.n, a.Lx, a.Ly, a.Lz, &p);

    while (!converged) {
        clear_forces(a.n, &p);
        converged = interactions(rc, a.Lx, a.Ly, a.Lz, a.n, &p);
        advance(dt, a.Lx, a.Ly, a.Lz, a.n, &p);
        ++ step;

        if (step > max_steps) {
            fprintf(stderr, "reached max steps\n");
            break;
        }

        if (step % 50 == 0) {
            char name[FILENAME_MAX];
            int id = step / 50;
            sprintf(name, "%.04d.xyz", id);
            dump_xyz(name, a.n, &p);
        }
    }

    dump_xyz("final.xyz", a.n, &p);
    
    particles_fin(&p);
        
    return 0;
}
