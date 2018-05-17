#include <stdio.h>
#include <string.h>

#include "utils/error.h"
#include "conf/imp.h"
#include "io/mesh_read/imp.h"

#include "imp.h"

/* ns: namespace */
static void get_desc(const char *ns, const char *d, char *desc) {
    sprintf(desc, "%s.%s", ns, d);
}

static void cfg_bool(const Config *c, const char *ns, const char *d, int *a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_bool(c, desc, a));
}

static void cfg_string(const Config *c, const char *ns, const char *d, const char **a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_string(c, desc, a));
}

static void cfg_int(const Config *c, const char *ns, const char *d, int *a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_int(c, desc, a));
}

static void cfg_float(const Config *c, const char *ns, const char *d, float *a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_float(c, desc, a));
}

#define name "rbc"

void rbc_force_set_conf(const MeshRead *cell, const Config *cfg, RbcForce *f) {
    int stress_free, rnd;
    UC(cfg_bool(cfg, name, "stress_free", &stress_free));
    UC(cfg_bool(cfg, name, "rnd", &rnd));

    if (stress_free) {
        const char *fname;
        UC(cfg_string(cfg, name, "stress_free_file", &fname));
        UC(rbc_force_set_stressfree(fname, /**/ f));
    }
    else {
        float Atot = 0;
        int nt;
        nt = mesh_read_get_nt(cell);
        UC(cfg_float(cfg, name, "totArea", &Atot));
        UC(rbc_force_set_stressful(nt, Atot, /**/ f));
    }

    if (rnd) {
        int seed;
        UC(cfg_int(cfg, name, "seed", &seed));
        UC(rbc_force_set_rnd1(seed, f));
    }
    else {
        UC(rbc_force_set_rnd0(f));
    }
}
