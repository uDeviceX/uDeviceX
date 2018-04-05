#include <stdio.h>

#include "utils/error.h"
#include "utils/imp.h"
#include "conf/imp.h"
#include "imp.h"

void den_map_set_conf(const Config *cfg, const Coords *c, DContMap *m) {
    const char *type;

    UC(conf_lookup_string(cfg, "denoutflow.type", &type));
    if (same_str(type, "none")) {
        UC(den_map_set_none(c, /**/ m));
    }
    else if (same_str(type, "circle")) {
        float R;
        UC(conf_lookup_float(cfg, "denoutflow.R", &R));
        UC(den_map_set_circle(c, R, /**/ m));
    } else {
        ERR("Unrecognized type <%s>", type);
    }
}
