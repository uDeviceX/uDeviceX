#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "parser/imp.h"

#include "imp.h"

void rbc_params_set_conf(const Config *c, RbcParams *par) {
    float gc, gt, kbt, kb, phi, ks, mpow, x0, ka, kd, kv, totArea, totVolume;

    UC(conf_lookup_float(c, "rbc.gammaC", &gc));
    UC(conf_lookup_float(c, "rbc.gammaT", &gt));
    UC(conf_lookup_float(c, "rbc.kBT", &kbt));
    UC(conf_lookup_float(c, "rbc.kb", &kb));
    UC(conf_lookup_float(c, "rbc.phi", &phi));
    UC(conf_lookup_float(c, "rbc.ks", &ks));
    UC(conf_lookup_float(c, "rbc.x0", &x0));
    UC(conf_lookup_float(c, "rbc.mpow", &mpow));
    UC(conf_lookup_float(c, "rbc.ka", &ka));
    UC(conf_lookup_float(c, "rbc.kd", &kd));
    UC(conf_lookup_float(c, "rbc.kv", &kv));

    UC(conf_lookup_float(c, "rbc.totArea",   &totArea));
    UC(conf_lookup_float(c, "rbc.totVolume", &totVolume));

    rbc_params_set_fluct(gc, gt, kbt, /**/ par);
    rbc_params_set_bending(kb, phi, /**/ par);
    rbc_params_set_spring(ks, x0, mpow, /**/ par);
    rbc_params_set_area_volume(ka, kd, kv, /**/ par);
    rbc_params_set_tot_area_volume(totArea, totVolume, /**/ par);
}
