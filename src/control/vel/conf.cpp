#include <stdio.h>
#include <mpi.h>
#include <vector_types.h>

#include "utils/imp.h"
#include "utils/error.h"
#include "parser/imp.h"

#include "imp.h"

void vcont_set_conf(const Config *cfg, /**/ PidVCont *c) {
    const char *type;
    float3 U;
    float factor, Kp, Ki, Kd;

    UC(conf_lookup_string(cfg, "vcon.type", &type));
    UC(conf_lookup_float3(cfg, "vcon.U", &U));
    UC(conf_lookup_float(cfg, "vcon.factor", &factor));
    UC(conf_lookup_float(cfg, "vcon.Kp", &Kp));
    UC(conf_lookup_float(cfg, "vcon.Ki", &Ki));
    UC(conf_lookup_float(cfg, "vcon.Kd", &Kd));

    UC(vcont_set_params(factor, Kp, Ki, Kd, /**/ c));
    UC(vcont_set_target(U, /**/ c));
    
    if      (same_str(type, "cart"))
        UC(vcont_set_cart(/**/ c));
    else if (same_str(type, "rad"))
        UC(vcont_set_radial(/**/ c));
    else
        ERR("Unrecognised type <%s>", type);

}
