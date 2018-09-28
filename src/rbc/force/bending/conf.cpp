#include <stdio.h>
#include <string.h>

#include "utils/error.h"
#include "conf/imp.h"
#include "io/mesh_read/imp.h"

#include "imp.h"

void rbc_bending_set_conf(const MeshRead *cell, const Config *cfg, const char *name, RbcForce *f) {
    int stress_free, rnd;
    UC(conf_lookup_bool_ns(cfg, name, "stress_free", &stress_free));
    UC(conf_lookup_bool_ns(cfg, name, "rnd", &rnd));

    if (stress_free) {
        const char *fname;
        UC(conf_lookup_string_ns(cfg, name, "stress_free_file", &fname));
        UC(rbc_bending_set_stressfree(fname, /**/ f));
    }
    else {
        float Atot = 0;
        int nt;
        nt = mesh_read_get_nt(cell);
        UC(conf_lookup_float_ns(cfg, name, "totArea", &Atot));
        UC(rbc_bending_set_stressful(nt, Atot, /**/ f));
    }
}
