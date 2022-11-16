#include <stdlib.h>
#include <stdio.h>
#include <vector_types.h>

#include "conf/imp.h"
#include "utils/imp.h"
#include "utils/error.h"

#include "imp.h"

void bforce_set_conf(const Config *cfg, /**/ BForce *bf) {
    const char *type;
    UC(conf_lookup_string(cfg, "bforce.type", /**/ &type));

    if      (same_str(type, "none")) {
        bforce_set_none(/**/ bf);
    }
    else if (same_str(type, "constant")) {
        float3 f;
        UC(conf_lookup_float3(cfg, "bforce.f", /**/ &f));
        UC(bforce_set_cste(f, /**/ bf));
    }
    else if (same_str(type, "double_poiseuille")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_set_dp(a, /**/ bf));
    }
    else if (same_str(type, "shear")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_set_shear(a, /**/ bf));
    }
    else if (same_str(type, "four_roller")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_set_rol(a, /**/ bf));
    }
    else if (same_str(type, "rad")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_set_rad(a, /**/ bf));
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}

