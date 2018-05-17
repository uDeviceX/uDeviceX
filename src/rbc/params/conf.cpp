#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "conf/imp.h"

#include "imp.h"

#define name "rbc"

void rbc_params_set_conf(const Config *c, RbcParams *par) {
    float gc, gt, kbt, kb, phi, ks, mpow, x0, ka, kd, kv, totArea, totVolume;

    UC(conf_lookup_float_ns(c, name, "gammaC", &gc));
    UC(conf_lookup_float_ns(c, name, "gammaT", &gt));
    UC(conf_lookup_float_ns(c, name, "kBT", &kbt));
    UC(conf_lookup_float_ns(c, name, "kb", &kb));
    UC(conf_lookup_float_ns(c, name, "phi", &phi));
    UC(conf_lookup_float_ns(c, name, "ks", &ks));
    UC(conf_lookup_float_ns(c, name, "x0", &x0));
    UC(conf_lookup_float_ns(c, name, "mpow", &mpow));
    UC(conf_lookup_float_ns(c, name, "ka", &ka));
    UC(conf_lookup_float_ns(c, name, "kd", &kd));
    UC(conf_lookup_float_ns(c, name, "kv", &kv));

    UC(conf_lookup_float_ns(c, name, "totArea",   &totArea));
    UC(conf_lookup_float_ns(c, name, "totVolume", &totVolume));

    rbc_params_set_fluct(gc, gt, kbt, /**/ par);
    rbc_params_set_bending(kb, phi, /**/ par);
    rbc_params_set_spring(ks, x0, mpow, /**/ par);
    rbc_params_set_area_volume(ka, kd, kv, /**/ par);
    rbc_params_set_tot_area_volume(totArea, totVolume, /**/ par);
}
