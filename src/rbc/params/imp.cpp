#include <stdlib.h>
#include <stdio.h>

#include "utils/imp.h"
#include "utils/error.h"

#include "type.h"
#include "imp.h"

void rbc_params_ini(RbcParams **p) {
    UC(emalloc(sizeof(RbcParams), (void**) p));
}

void rbc_params_fin(RbcParams *p) {
    UC(efree(p));
}

void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *par) {
    par->gammaC = gammaC;
    par->gammaT = gammaT;
    par->kBT = kBT;
}

void rbc_params_set_bending(float kb, float phi, RbcParams *par) {
    par->kb = kb;
    par->phi = phi;
}

void rbc_params_set_spring(float ks, float x0, float mpow, RbcParams *par) {
    par->ks = ks;
    par->x0 = x0;
    par->mpow = mpow;
}

void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *par) {
    par->totArea = totArea;
    par->totVolume = totVolume;
}

void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *par) {
    par->ka = ka;
    par->kv = kv;
    par->kd = kd;
}

double rbc_params_get_tot_volume(const RbcParams *par) {
    return par->totVolume;
}

RbcParams_v rbc_params_get_view(const RbcParams *p) {
    RbcParams_v v;
    v.gammaC = p->gammaC;
    v.gammaT = p->gammaT;
    v.kBT = p->kBT;
    v.kb = p->kb;
    v.phi = p->phi;
    v.ks = p->ks;
    v.x0 = p->x0;
    v.mpow = p->mpow;
    v.ka = p->ka;
    v.kd = p->kd;
    v.kv = p->kv;
    v.totArea = p->totArea;
    v.totVolume = p->totVolume;
    return v;
}
