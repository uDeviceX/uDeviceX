#include <stdlib.h>
#include <stdio.h>
#include <vector_types.h>

#include "glob/type.h"
#include "parser/imp.h"
#include "utils/imp.h"
#include "utils/error.h"

#include "imp.h"
#include "conf.h"

void bforce_ini_conf(const Config *cfg, /**/ BForce *bf) {
    const char *type;
    UC(conf_lookup_string(cfg, "bforce.type", /**/ &type));

    if      (same_str(type, "none")) {
        bforce_ini_none(/**/ bf);
    }
    else if (same_str(type, "constant")) {
        float3 f;
        UC(conf_lookup_float3(cfg, "bforce.f", /**/ &f));
        UC(bforce_ini_cste(f, /**/ bf));
    }
    else if (same_str(type, "double_poiseuille")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_dp(a, /**/ bf));
    }
    else if (same_str(type, "shear")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_shear(a, /**/ bf));
    }
    else if (same_str(type, "four_roller")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_rol(a, /**/ bf));
    }
    else if (same_str(type, "rad")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_rad(a, /**/ bf));
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}

