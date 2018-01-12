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

void rbc_params_set(RbcParams *p) {

}

RbcParams_v rbc_params_get_view(const RbcParams *p) {
    RbcParams_v v;
    v.gammaC = p->gammaC;
    v.kBT = p->kBT;
    v.kb = p->kb;
    v.phi = p->phi;
    v.Cp = p->Cp;
    v.x0 = p->x0;
    v.ka = p->ka;
    v.kd = p->kd;
    v.kv = p->kv;
    return v;
}
