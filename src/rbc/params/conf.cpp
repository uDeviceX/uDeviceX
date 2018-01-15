#include <stdio.h>

#include "utils/error.h"
#include "parser/imp.h"

#include "imp.h"

void rbc_params_set_conf(const Config *c, RbcParams *p) {
    float gc, kbt, kb, phi, cp, x0, ka, kd, kv;

    UC(conf_lookup_float(c, "rbc.gammaC", &gc));
    UC(conf_lookup_float(c, "rbc.kBT", &kbt));
    UC(conf_lookup_float(c, "rbc.kb", &kb));
    UC(conf_lookup_float(c, "rbc.phi", &phi));
    UC(conf_lookup_float(c, "rbc.Cp", &cp));
    UC(conf_lookup_float(c, "rbc.x0", &x0));
    UC(conf_lookup_float(c, "rbc.ka", &ka));
    UC(conf_lookup_float(c, "rbc.kd", &kd));
    UC(conf_lookup_float(c, "rbc.kv", &kv));
    
    rbc_params_set_fluct(gc, kbt, /**/ p);
    rbc_params_set_bending(kb, phi, /**/ p);
    rbc_params_set_spring(cp, x0, /**/ p);
    rbc_params_set_area_volume(ka, kd, kv, /**/ p);
}
