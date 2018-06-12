#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "type.h"
#include "utils.h"
#include "particles.h"
#include "matrices.h"

static void usg() {
    fprintf(stderr,
            "usage: u.obj-ic [-d] <N> <Lx> <Ly> <Lz> <r> <sc> [RND, <ax> <ay> <az>]\n"
            "\t N          : number of spheres to pack\n"
            "\t Lx, Ly, Lz : dimension of domain\n"
            "\t sc         : scale\n"
            "\n"
            "\t RND        : random angles\n"
            "\t or\n"
            "\t ax ay az   : given angles\n"
            "\t -d         : optional, dumps xyz coords\n");
    exit(1);
}

static bool shift(int *c, char ***v) {
    (*c) --; (*v)++;
    return (*c) > 0;
}

static void parse(int c, char **v, Args *a) {
    if (!shift(&c, &v)) usg();

    a->dump_xyz = false;
    if (0 == strcmp("-d", *v)) {
        a->dump_xyz = true;
        if (!shift(&c, &v)) usg();
    }
    
    a->n = atoi(*v);

    if (!shift(&c, &v)) usg();
    a->Lx = atoi(*v);
    if (!shift(&c, &v)) usg();
    a->Ly = atoi(*v);
    if (!shift(&c, &v)) usg();
    a->Lz = atoi(*v);

    if (!shift(&c, &v)) usg();
    a->r = atof(*v);
    if (!shift(&c, &v)) usg();
    a->sc = atof(*v);

    if (!shift(&c, &v)) usg();
    if (0 == strcmp(*v, "RND")) {
        a->ang.type = ANGLE_RND;
    }
    else {
        a->ang.type = ANGLE_IN;
        a->ang.x = atof(*v);
        if (!shift(&c, &v)) usg();
        a->ang.y = atof(*v);
        if (!shift(&c, &v)) usg();
        a->ang.z = atof(*v);
    }    
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
    fprintf(stderr, "max sphere density: %g\n", den);
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
        
        if (step > max_steps) {
            fprintf(stderr, "reached max steps\n");
            break;
        }

        if (a.dump_xyz && (step % freq == 0)) {
            char name[FILENAME_MAX];
            int id = step / freq;
            sprintf(name, "%.04d.xyz", id);
            dump_xyz(name, a.n, &p);
        }
    }

    if (a.dump_xyz)
        dump_xyz("final.xyz", a.n, &p);
    
    particles_fin(&p);

    if (step < max_steps)
        fprintf(stderr, "Done in %d iterations\n", step);
    else
        return 1;

    particles_shift(a.Lx/2, a.Ly/2, a.Lz/2, a.n, &p);
    dump_matrices(a.sc, &a.ang, a.n, &p, stdout);
    
    return 0;
}
