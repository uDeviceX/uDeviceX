#include <stdio.h>
#include <mpi.h>

#include "utils/msg.h"
#include "mpi/glb.h"
#include "utils/error.h"
#include "parser/imp.h"

enum {MAX_VEC=128};

static void extract(const Config *c) {
    int i, a, opt, in, fn, ivec[MAX_VEC];
    float f, fvec[MAX_VEC];

    UC(conf_lookup_int(c, "a", &a));
    UC(conf_lookup_float(c, "f", &f));

    printf("%d %g\n", a, f);

    if (conf_opt_int(c, "opt", &opt))
        printf("%d\n", opt);

    if (conf_opt_vint(c, "ivec", &in, ivec)) {
        for (i = 0; i < in; ++i)
            printf("%d ", ivec[i]);
        printf("\n");
    }

    if (conf_opt_vfloat(c, "fvec", &fn, fvec)) {
        for (i = 0; i < fn; ++i)
            printf("%g ", fvec[i]);
        printf("\n");
    }
}

int main(int argc, char **argv) {
    Config *cfg;
    int dims[3];

    m::ini(&argc, &argv);
    // eat executable and dims
    m::get_dims(&argc, &argv, dims);

    conf_ini(/**/ &cfg);
    conf_read(argc, argv, /**/ cfg);

    UC(extract(cfg));
    
    conf_fin(/**/ cfg);    

    m::fin();
}
