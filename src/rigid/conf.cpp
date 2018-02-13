#include <stdio.h>
#include <vector_types.h>

#include "utils/error.h"
#include "parser/imp.h"

#include "imp.h"

void rig_set_pininfo_conf(const Config *cfg, RigPinInfo *pi) {
    int3 com, axis;

    UC(conf_lookup_int3(cfg, "rig.pin_com", &com));
    UC(conf_lookup_int3(cfg, "rig.pin_axis", &axis));
    
    UC(rig_set_pininfo(com, axis, pi));
}
