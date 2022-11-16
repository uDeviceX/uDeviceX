#include <stdio.h>
#include <math.h>
#include <string.h>

#include "utils/error.h"
#include "conf/imp.h"

#include "inc/def.h"
#include "type.h"
#include "imp.h"

static int get_ncol(int npar) {
    int d = lround( sqrt(1 + 8 * npar) );
    return (d - 1) / 2;
}

void pair_set_conf(const Config *cfg, const char *ns, PairParams *par) {
    int dpd, lj, adhesion;

    UC(conf_lookup_bool_ns(cfg, ns, "dpd",      &dpd));
    UC(conf_lookup_bool_ns(cfg, ns, "lj",       &lj));
    UC(conf_lookup_bool_ns(cfg, ns, "adhesion", &adhesion));

    if (dpd) {
        int na, ng, nc;
        float a[MAX_PAR], g[MAX_PAR], spow;

        UC(conf_lookup_vfloat_ns(cfg, ns, "a", MAX_PAR, &na, a));
        UC(conf_lookup_vfloat_ns(cfg, ns, "g", MAX_PAR, &ng, g));
        UC(conf_lookup_float_ns(cfg, ns, "spow", &spow));

        if (na != ng)
            ERR("%s.a and %s.g must have the same length: %d / %d", ns, ns, na, ng);
        
        nc = get_ncol(na);
        
        UC(pair_set_dpd(nc, a, g, spow, /**/ par));
    }

    if (lj) {
        float s, e;

        UC(conf_lookup_float_ns(cfg, ns, "ljs", &s));
        UC(conf_lookup_float_ns(cfg, ns, "lje", &e));

        UC(pair_set_lj(s, e, /**/ par));
    }

    if (adhesion) {
        float k1, k2;

        UC(conf_lookup_float_ns(cfg, ns, "k1", &k1));
        UC(conf_lookup_float_ns(cfg, ns, "k2", &k2));

        UC(pair_set_adhesion(k1, k2, /**/ par));
    }
}
