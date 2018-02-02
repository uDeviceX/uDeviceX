#include <stdio.h>
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

void pair_set_conf(const Config *cfg, const char *base, PairParams *par) {
    int dpd, lj;
    char desc[FILENAME_MAX];

    get_desc(base, "dpd", desc);
    UC(conf_lookup_bool(cfg, desc, &dpd));

    get_desc(base, "lj", desc);
    UC(conf_lookup_bool(cfg, desc, &lj));

    if (dpd) {
        
    }
    if (lj) {

    }
}
