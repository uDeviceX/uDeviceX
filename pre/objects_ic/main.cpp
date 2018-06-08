#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <math.h>

#include "type.h"
#include "utils.h"
#include "particles.h"

static void usg() {
    fprintf(stderr,
            "usage: obj-ic <N> <Lx> <Ly> <Lz> <r>\n"
            "\t N          : number of spheres to pack\n"
            "\t Lx, Ly, Lz : dimension of domain\n");
    exit(1);
}

static bool shift(int *c, char ***v) {
    (*c) --; (*v)++;
    return (*c) > 0;
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

    if (!shift(&c, &v)) usg();
    a->r = atof(*v);
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

static void check_density(Args a) {
    real den, Vs, Vb;
    Vb = a.Lx * a.Ly * a.Lz;
    Vs = a.n * 4.0 * M_PI * cu(a.r) / 3.0;
    den = Vs / Vb;
    fprintf(stderr, "%g\n", den);
}

int main(int argc, char **argv) {
    Args a;
    Particles p;
    long seed = 12345;
    bool converged = false;
    int step = 0;
    real rmin, T, T0;

    static const real dt = 0.1;
    static const int max_steps = 15000;
    static const int freq = 100;

    parse(argc, argv, &a);
    check_density(a);
    srand(seed);
    
    particles_ini(a.n, a.Lx, a.Ly, a.Lz, &p);

    rmin = a.r;

    while (!converged) {
        particles_clear_forces(a.n, &p);
        rmin = particles_interactions(2.5 * a.r, a.Lx, a.Ly, a.Lz, a.n, &p);
        particles_advance(dt, a.Lx, a.Ly, a.Lz, a.n, &p);
        ++ step;

        T = particles_temperature(a.n, &p);
        T0 = 0.1 / (sqrt(step + 10));
        particles_rescale_v(T0, T, a.n, &p);
        
        converged = (a.r < rmin / 2);
        printf("%g\n", rmin / 2);
        
        if (step > max_steps) {
            fprintf(stderr, "reached max steps\n");
            break;
        }

        if (step % freq == 0) {
            
            char name[FILENAME_MAX];
            int id = step / freq;
            sprintf(name, "%.04d.xyz", id);
            dump_xyz(name, a.n, &p);
        }
    }

    dump_xyz("final.xyz", a.n, &p);
    
    particles_fin(&p);

    if (step < max_steps)
        fprintf(stderr, "Done in %d iterations\n", step);
    else
        return 1;

    return 0;
}
