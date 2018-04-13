#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/ini.h"
#include "coords/imp.h"

#include "utils/msg.h"
#include "mpi/glb.h"

#include "utils/error.h"
#include "utils/mc.h"
#include "utils/imp.h"

#include "conf/imp.h"

#include "io/grid/imp.h"

#include "mpi/wrapper.h"

typedef float (*Func)(float, float, float);

static float f1(float x, float y, float z) {
    return x*x + 2 * y - x*z;
}

static float f2(float x, float y, float z) {
    return 2.0 - x*x + z * y - x*z;
}

static void fill_data(const Coords *g, Func f, float *a) {
    int Lx, Ly, Lz, i, j, k, id;
    int ox, oy, oz, lx, ly, lz;;
    float x, y, z;
    Lx = xdomain(g); ox = xlo(g); lx = xs(g);
    Ly = ydomain(g); oy = ylo(g); ly = ys(g);
    Lz = zdomain(g); oz = zlo(g); lz = zs(g);

    for (id = k = 0; k < lz; ++k) {
        for (j = 0; j < ly; ++j) {
            for (i = 0; i < lx; ++i, ++id) {
                x = (float) (ox + i) / (float) Lx;
                y = (float) (oy + j) / (float) Ly;
                z = (float) (oz + k) / (float) Lz;
                a[id] = f(x, y, z);
            }
        }
    }
}


static void dump(MPI_Comm cart, const Coords *l, const Coords *g, const char *path) {
    enum {NCMP = 2};
    size_t nc;
    int i;
    int3 L, N, s;
    const char *names[NCMP] = { "f1", "f2" };
    const Func ff[NCMP] = {f1, f2};
    float *data[NCMP];

    s.x = xs(g); N.x = xdomain(g); L.x = xdomain(l);
    s.y = ys(g); N.y = ydomain(g); L.y = ydomain(l);
    s.z = zs(g); N.z = zdomain(g); L.z = zdomain(l);
        
    nc = s.x * s.y * s.z;

    for (i = 0; i < NCMP; ++i) {
        EMALLOC(nc, &data[i]);
        fill_data(g, ff[i], data[i]);
    }

    UC(grid_write(N, L, cart, path, NCMP, (const float**) data, names));

    for (i = 0; i < NCMP; ++i) EFREE(data[i]);
}

int main(int argc, char **argv) {
    const char *dir;
    char path[FILENAME_MAX];
    Coords *l, *g; // space and grid coordinates
    Config *cfg;
    int rank, dims[3];
    int3 N; // local grid size
    MPI_Comm cart;
    
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    m::Comm_rank(cart, &rank);
    msg_ini(rank);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_int3(cfg, "grid_size", &N));
    UC(conf_lookup_string(cfg, "dir", &dir));
    
    UC(coords_ini_conf(cart, cfg, &l));
    UC(coords_ini(cart, N.x, N.y, N.z, &g));

    sprintf(path, DUMP_BASE "/%s/test.h5", dir);
    dump(cart, l, g, path);
    
    UC(coords_fin(l));
    UC(coords_fin(g));
    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
}
