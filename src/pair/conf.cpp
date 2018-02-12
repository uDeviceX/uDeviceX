#include <stdio.h>
#include <math.h>
#include <string.h>

#include "utils/error.h"
#include "parser/imp.h"

#include "type.h"
#include "imp.h"

/* set string "d" = "base.var" */
static void get_desc(const char *base, const char *var, char *d) {
    strcpy(d, base);
    strcat(d, ".");
    strcat(d, var);
}

static int get_ncol(int npar) {
    int d = lround( sqrt(1 + 8 * npar) );
    return (d - 1) / 2;
}

void pair_set_conf(const Config *cfg, const char *base, PairParams *par) {
    ERR("pair_set_conf: shell not be used");

    int dpd, lj;
    char desc[FILENAME_MAX];

    get_desc(base, "dpd", desc);
    UC(conf_lookup_bool(cfg, desc, &dpd));

    get_desc(base, "lj", desc);
    UC(conf_lookup_bool(cfg, desc, &lj));

    if (dpd) {
        int na, ng, nc;
        float a[MAX_PAR], g[MAX_PAR];

        get_desc(base, "a", desc);
        UC(conf_lookup_vfloat(cfg, desc, &na, a));

        if (na > MAX_PAR)
            ERR("Too many parameters in %s.a : %d/%d", base, na, MAX_PAR);
            
        get_desc(base, "g", desc);
        UC(conf_lookup_vfloat(cfg, desc, &ng, g));

        if (na != ng)
            ERR("%s.a and %s.g must have the same length: %d / %d", base, base, na, ng);
        
        nc = get_ncol(na);
        
        UC(pair_set_dpd(nc, a, g, /**/ par));
    }
    if (lj) {
        float s, e;

        get_desc(base, "ljs", desc);
        UC(conf_lookup_float(cfg, desc, &s));

        get_desc(base, "lje", desc);
        UC(conf_lookup_float(cfg, desc, &e));

        UC(pair_set_lj(s, e, /**/ par));
    }
}
