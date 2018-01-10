#include <stdlib.h>
#include <stdio.h>
#include <vector_types.h>

#include "glob/type.h"
#include "parser/imp.h"
#include "utils/imp.h"
#include "utils/error.h"

#include "imp.h"

void inflow_ini_conf(Coords coords, const Config *cfg, /**/ Inflow *i) {
    const char *type;
    UC(conf_lookup_string(cfg, "outflow.type", &type));
    
    if      (same_str(type, "circle")) {
        float R, H, U;
        int poiseuille;
        float3 center;

        UC(conf_lookup_float(cfg, "inflow.R", &R));
        UC(conf_lookup_float(cfg, "inflow.H", &H));
        UC(conf_lookup_float(cfg, "inflow.U", &U));
        UC(conf_lookup_bool(cfg, "inflow.poiseuille", &poiseuille));
        UC(conf_lookup_float3(cfg, "inflow.center", &center));        
        UC(inflow_ini_params_circle(coords, center, R, H, U, poiseuille, /**/ i));
    }
    else if (same_str(type, "plate")) {
        int upois, vpois, dir;
        float3 origin, u;
        float L1, L2;

        UC(conf_lookup_float(cfg, "inflow.L1", &L1));
        UC(conf_lookup_float(cfg, "inflow.L2", &L2));
        UC(conf_lookup_int(cfg, "inflow.direction", &dir));
        UC(conf_lookup_bool(cfg, "inflow.upoiseuille", &upois));
        UC(conf_lookup_bool(cfg, "inflow.vpoiseuille", &vpois));
        UC(conf_lookup_float3(cfg, "inflow.origin", &origin));
        UC(conf_lookup_float3(cfg, "inflow.u", &u));
        
        UC(inflow_ini_params_plate(coords, origin, dir, L1, L2, u, upois, vpois, /**/ i));
    }
    else {
        ERR("unknown inflow type <%s>", type);
    }
}

