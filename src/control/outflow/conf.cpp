#include <stdio.h>
#include <vector_types.h>

#include "utils/error.h"
#include "utils/imp.h"
#include "conf/imp.h"

#include "imp.h"

void outflow_set_cfg(const Config *cfg, const Coords *coords, Outflow *o) {
    const char *type;
    UC(conf_lookup_string(cfg, "outflow.type", &type));

    if      (same_str(type, "circle")) {
        float3 center;
        float R;
        UC(conf_lookup_float(cfg, "outflow.R", &R));
        UC(conf_lookup_float3(cfg, "outflow.center", &center));

        outflow_set_params_circle(coords, center, R, /**/ o);
    }
    else if (same_str(type, "plate")) {
        int dir;
        float r0;
        UC(conf_lookup_int(cfg, "outflow.direction", &dir));
        UC(conf_lookup_float(cfg, "outflow.position", &r0));
        outflow_set_params_plate(coords, dir, r0, /**/ o);
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}
