#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "conf/imp.h"

#include "imp.h"

/* ns: namespace */
static void get_desc(const char *ns, const char *d, char *desc) {
    sprintf(desc, "%s.%s", ns, d);
}

static void cfg_float(const Config *c, const char *ns, const char *d, float *a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_float(c, desc, a));
}

#define name "rbc"

void rbc_params_set_conf(const Config *c, RbcParams *par) {
    float gc, gt, kbt, kb, phi, ks, mpow, x0, ka, kd, kv, totArea, totVolume;

    UC(cfg_float(c, name, "gammaC", &gc));
    UC(cfg_float(c, name, "gammaT", &gt));
    UC(cfg_float(c, name, "kBT", &kbt));
    UC(cfg_float(c, name, "kb", &kb));
    UC(cfg_float(c, name, "phi", &phi));
    UC(cfg_float(c, name, "ks", &ks));
    UC(cfg_float(c, name, "x0", &x0));
    UC(cfg_float(c, name, "mpow", &mpow));
    UC(cfg_float(c, name, "ka", &ka));
    UC(cfg_float(c, name, "kd", &kd));
    UC(cfg_float(c, name, "kv", &kv));

    UC(cfg_float(c, name, "totArea",   &totArea));
    UC(cfg_float(c, name, "totVolume", &totVolume));

    rbc_params_set_fluct(gc, gt, kbt, /**/ par);
    rbc_params_set_bending(kb, phi, /**/ par);
    rbc_params_set_spring(ks, x0, mpow, /**/ par);
    rbc_params_set_area_volume(ka, kd, kv, /**/ par);
    rbc_params_set_tot_area_volume(totArea, totVolume, /**/ par);
}
