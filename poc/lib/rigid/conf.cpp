#include <stdio.h>
#include <vector_types.h>

#include "utils/error.h"
#include "conf/imp.h"

#include "imp.h"

void rig_pininfo_set_conf(const Config *cfg, const char *ns, RigPinInfo *pi) {
    int3 com, axis;
    int pdir;

    UC(conf_lookup_int3_ns(cfg, ns, "pin_com", &com));
    UC(conf_lookup_int3_ns(cfg, ns, "pin_axis", &axis));
    UC(conf_lookup_int_ns(cfg, ns, "pdir", &pdir));

    UC(rig_pininfo_set(com, axis, pi));
    UC(rig_pininfo_set_pdir(pdir, pi));
}
