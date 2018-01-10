#include "parser/imp.h"

#include "imp.h"
#include "conf.h"

void bforce_ini_conf(const Config *cfg, /**/ BForce *bf) {
    const char *type;
    UC(conf_lookup_string(cfg, "bforce.type", /**/ &type));

    if      (same_str(type, "none")) {
        bforce_ini_none(/**/ bforce);
    }
    else if (same_str(type, "constant")) {
        float3 f;
        UC(conf_lookup_float3(cfg, "bforce.f", /**/ &f));
        UC(bforce_ini_cste(f, /**/ bforce));
    }
    else if (same_str(type, "double_poiseuille")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_dp(a, /**/ bforce));
    }
    else if (same_str(type, "shear")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_shear(a, /**/ bforce));
    }
    else if (same_str(type, "four_roller")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_rol(a, /**/ bforce));
    }
    else if (same_str(type, "rad")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_rad(a, /**/ bforce));
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}

