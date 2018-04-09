#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "conf/imp.h"
#include "utils/error.h"
#include "utils/imp.h"

#include "io/mesh/imp.h"

#include "imp.h"

static int get_shifttype(const Config *c, const char *desc) {
    const char *type;
    UC(conf_lookup_string(c, desc, &type));
    if      (same_str(type, "edge"  )) return EDGE;
    else if (same_str(type, "center")) return CENTER;
    else
        ERR("Unrecognised rbc shift type <%s>", type);
    return -1;
}

void opt_read_common(const Config *c, Opt *o) {
    int b;
    const char *s;

    UC(conf_lookup_bool(c, "flu.ids", &b));
    o->fluids = b;

    UC(conf_lookup_string(c, "dump.strt_base_dump", &s));
    strcpy(o->strt_base_dump, s);

    UC(conf_lookup_string(c, "dump.strt_base_read", &s));
    strcpy(o->strt_base_read, s);

    UC(conf_lookup_bool(c, "dump.field", &b));
    o->dump_field = b;
    UC(conf_lookup_float(c, "dump.freq_field", &o->freq_field));
    
    UC(conf_lookup_bool(c, "dump.strt", &b));
    o->dump_strt = b;
    UC(conf_lookup_float(c, "dump.freq_strt", &o->freq_strt));

    UC(conf_lookup_bool(c, "dump.parts", &b));
    o->dump_parts = b;
    UC(conf_lookup_float(c, "dump.freq_parts", &o->freq_parts));
}

void opt_set_eq(Opt *o) {
    o->rbc  = false;
    o->rig  = false;
    o->wall = false;

    o->fsi = false;
    o->cnt = false;

    o->inflow = false;
    o->outflow = false;
    o->denoutflow = false;
    o->outflow = false;
    o->vcon = false;
    
    o->push_flu = false;
    o->push_rbc = false;
    o->push_rig = false;
}

void opt_read(const Config *c, Opt *o) {
    int b;
    UC(conf_lookup_bool(c, "fsi.active", &b));
    o->fsi = b;
    UC(conf_lookup_bool(c, "cnt.active", &b));
    o->cnt = b;

    UC(conf_lookup_bool(c, "flu.colors", &b));
    o->flucolors = b;
    UC(conf_lookup_bool(c, "flu.ids", &b));
    o->fluids = b;
    UC(conf_lookup_bool(c, "flu.stresses", &b));
    o->fluss = b;

    UC(conf_lookup_bool(c, "rbc.active", &b));
    o->rbc = b;
    UC(conf_lookup_bool(c, "rbc.ids", &b));
    o->rbcids = b;
    UC(conf_lookup_bool(c, "rbc.stretch", &b));
    o->rbcstretch = b;
    o->rbcshifttype = get_shifttype(c, "rbc.shifttype");

    UC(conf_lookup_bool(c, "rig.active", &b));
    o->rig = b;
    UC(conf_lookup_bool(c, "rig.bounce", &b));
    o->rig_bounce = b;
    UC(conf_lookup_bool(c, "rig.empty_pp", &b));
    o->rig_empty_pp = b;
    o->rigshifttype = get_shifttype(c, "rig.shifttype");

    UC(conf_lookup_bool(c, "wall.active", &b));
    o->wall = b;
    
    UC(conf_lookup_bool(c, "outflow.active", &b));
    o->outflow = b;
    UC(conf_lookup_bool(c, "inflow.active", &b));
    o->inflow = b;
    UC(conf_lookup_bool(c, "denoutflow.active", &b));
    o->denoutflow = b;
    UC(conf_lookup_bool(c, "vcon.active", &b));
    o->vcon = b;

    UC(conf_lookup_bool(c, "dump.rbc_com", &b));
    o->dump_rbc_com = b;
    UC(conf_lookup_float(c, "dump.freq_rbc_com", &o->freq_rbc_com));

    UC(conf_lookup_bool(c, "dump.forces", &b));
    o->dump_forces = b;
    
    UC(conf_lookup_int(c, "flu.recolor_freq", &o->recolor_freq));

    UC(conf_lookup_bool(c, "flu.push", &b));
    o->push_flu = b;
    UC(conf_lookup_bool(c, "rbc.push", &b));
    o->push_rbc = b;
    UC(conf_lookup_bool(c, "rig.push", &b));
    o->push_rig = b;
}
