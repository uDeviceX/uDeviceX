#include <stdio.h>
#include <mpi.h>

#include "utils/imp.h"
#include "utils/error.h"
#include "parser/imp.h"

#include "imp.h"

void scheme_restrain_set_conf(const Config *cfg, Restrain *r) {
    const char *kind;
    int freq;

    UC(conf_lookup_string(cfg, "restrain.kind", &kind));
    UC(conf_lookup_int(cfg, "restrain.freq", &freq));

    scheme_restrain_set_freq(0, r);
    
    if      (same_str(kind, "none"))
        scheme_restrain_set_none(r);
    else if (same_str(kind, "rbc"))
        scheme_restrain_set_rbc(r);
    else if (same_str(kind, "red"))
        scheme_restrain_set_red(r);
    else
        ERR("Unrecognise kind <%s>", kind);
}
