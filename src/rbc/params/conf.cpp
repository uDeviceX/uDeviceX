#include <stdio.h>

#include "utils/error.h"
#include "parser/imp.h"

#include "imp.h"

void rbc_params_set_conf(const Config *c, RbcParams *par) {
    float gc, gt, kbt, kb, phi, p, mpow, x0, ka, kd, kv;

    UC(conf_lookup_float(c, "rbc.gammaC", &gc));
    UC(conf_lookup_float(c, "rbc.gammaT", &gt));
    UC(conf_lookup_float(c, "rbc.kBT", &kbt));
    UC(conf_lookup_float(c, "rbc.kb", &kb));
    UC(conf_lookup_float(c, "rbc.phi", &phi));
    UC(conf_lookup_float(c, "rbc.p", &p));
    UC(conf_lookup_float(c, "rbc.x0", &x0));
    UC(conf_lookup_float(c, "rbc.mpow", &mpow));
    UC(conf_lookup_float(c, "rbc.ka", &ka));
    UC(conf_lookup_float(c, "rbc.kd", &kd));
    UC(conf_lookup_float(c, "rbc.kv", &kv));
    
    rbc_params_set_fluct(gc, gt, kbt, /**/ par);
    rbc_params_set_bending(kb, phi, /**/ par);
    rbc_params_set_spring(p, x0, mpow, /**/ par);
    rbc_params_set_area_volume(ka, kd, kv, /**/ par);
}
