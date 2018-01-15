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

void rbc_params_set_fluct(float gammaC, float kBT, RbcParams *p) {
    p->gammaC = gammaC;
    p->kBT = kBT;
}

void rbc_params_set_bending(float kb, float phi, RbcParams *p) {
    p->kb = kb;
    p->phi = phi;
}

void rbc_params_set_spring(float Cp, float x0, RbcParams *p) {
    p->Cp = Cp;
    p->x0 = x0;
}

void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *p) {
    p->ka = ka;
    p->kv = kv;
    p->kd = kd;
}

RbcParams_v rbc_params_get_view(const RbcParams *p) {
    RbcParams_v v;
    v.gammaC = p->gammaC;
    v.kBT0 = p->kBT;
    v.kb = p->kb;
    v.phi = p->phi;
    v.Cp = p->Cp;
    v.x0 = p->x0;
    v.ka = p->ka;
    v.kd = p->kd;
    v.kv = p->kv;
    return v;
}
