#include <stdio.h>
#include <math.h>
#include <string.h>

/*only because of kBT and dt, TODO: remove */
#include <conf.h>
#include "inc/conf.h"

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

static void compute_sigma(int n, const float g[], float kBT0, float dt0, float s[]) {
    for (int i = 0; i < n; ++i)
        s[i] = sqrt (2 * kBT0 * g[i] / dt0);
}

void pair_set_conf(const Config *cfg, const char *base, PairParams *par) {
    int dpd, lj;
    char desc[FILENAME_MAX];

    get_desc(base, "dpd", desc);
    UC(conf_lookup_bool(cfg, desc, &dpd));

    get_desc(base, "lj", desc);
    UC(conf_lookup_bool(cfg, desc, &lj));

    if (dpd) {
        int na, ng;
        float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR], kBT0, dt0;

        get_desc(base, "a", desc);
        UC(conf_lookup_vfloat(cfg, desc, &na, a));

        if (na > MAX_PAR)
            ERR("Too many parameters in %s.a : %d/%d", base, na, MAX_PAR);
            
        get_desc(base, "g", desc);
        UC(conf_lookup_vfloat(cfg, desc, &ng, g));

        if (na != ng)
            ERR("%s.a and %s.g must have the same length: %d / %d", base, base, na, ng);

        // TODO read from conf
        kBT0 = kBT;
        dt0  = dt;
        
        compute_sigma(na, g, kBT0, dt0, /**/ s);

        UC(pair_set_dpd(na, a, g, s, /**/ par));
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
