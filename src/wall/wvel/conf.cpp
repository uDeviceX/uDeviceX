#include <stdio.h>
#include <vector_types.h>

#include <conf.h>

#include "utils/imp.h"
#include "utils/error.h"
#include "parser/imp.h"

#include "imp.h"

void wvel_set_conf(const Config *cfg, Wvel *vw) {
    const char *type;
    UC(conf_lookup_string(cfg, "wvel.type", &type));

    if (same_str(type, "constant")) {
        float3 u;
        UC(conf_lookup_float3(cfg, "wvel.u", &u));
        UC(wvel_set_cste(u, vw));
    }
    else if (same_str(type, "shear")) {
        float gdot;
        int vdir, gdir, half;
        UC(conf_lookup_float(cfg, "wvel.gdot", &gdot));
        UC(conf_lookup_int(cfg, "wvel.vdir", &vdir));
        UC(conf_lookup_int(cfg, "wvel.gdir", &gdir));
        UC(conf_lookup_int(cfg, "wvel.half", &half));
        UC(wvel_set_shear(gdot, vdir, gdir, half, vw));
    }
    else if (same_str(type, "shear sin")) {
        float gdot, w;
        int vdir, gdir, half, log_freq;
        UC(conf_lookup_float(cfg, "wvel.gdot", &gdot));
        UC(conf_lookup_int(cfg, "wvel.vdir", &vdir));
        UC(conf_lookup_int(cfg, "wvel.gdir", &gdir));
        UC(conf_lookup_int(cfg, "wvel.half", &half));
        UC(conf_lookup_int(cfg, "wvel.log_freq", &log_freq));
        UC(conf_lookup_float(cfg, "wvel.w", &w));
        UC(wvel_set_shear_sin(gdot, vdir, gdir, half, w, log_freq, vw));
    }
    else if (same_str(type, "hele shaw")) {
        float u, h;
        UC(conf_lookup_float(cfg, "wvel.u", &u));
        UC(conf_lookup_float(cfg, "wvel.h", &h));
        UC(wvel_set_hs(u, h, vw));
    }
    else {
        ERR("unknown type <%s>\n", type);
    }
    //UC(conf_lookup_float(cfg, "wvel.dt0", &dt0));
}
