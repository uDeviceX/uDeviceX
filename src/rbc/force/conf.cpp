#include <stdio.h>

#include "utils/error.h"
#include "parser/imp.h"

#include "imp.h"

void rbc_force_set_conf(const MeshRead *cell, const Config *cfg, RbcForce *f) {
    bool stress_free, rnd;
    UC(conf_lookup_bool(cfg, "rbc.stress_free", &stress_free));
    UC(conf_lookup_bool(cfg, "rbc.rnd", &rnd));

    if (stress_free) {
        const char *fname;
        UC(conf_lookup_bool(cfg, "rbc.stress_free_file", &fname));
        UC(rbc_force_set_stressfree(fname, /**/ f));
    }
    else {
        float Atot = 0;
        UC(conf_lookup_float(cfg, "rbc.totArea", &Atot));
        UC(rbc_force_set_stressful(nt, Atot, /**/ r->force));
    }

    if (rnd) {
        int seed;
        UC(conf_lookup_int(cfg, "rbc.seed", &seed));
        UC(rbc_force_set_rnd1(seed, r->force));
    }
    else {
        UC(rbc_force_set_rnd0(r->force));
    }
}
