#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

#include "conf/imp.h"
#include "utils/error.h"
#include "utils/imp.h"

#include "io/mesh/imp.h"

#include "imp.h"

static int get_shifttype(const Config *c, const char *desc) {
    const char *type;
    UC(conf_lookup_string(c, desc, &type));
    if      (same_str(type, "edge"  )) return MESH_WRITE_EDGE;
    else if (same_str(type, "center")) return MESH_WRITE_CENTER;
    else
        ERR("Unrecognised rbc shift type <%s>", type);
    return -1;
}

static void lookup_bool(const Config *c, const char *desc, bool *res) {
    int b;
    UC(conf_lookup_bool(c, desc, &b));
    *res = b;
}

static void lookup_string(const Config *c, const char *desc, char *res) {
    const char *s;
    UC(conf_lookup_string(c, desc, &s));
    strcpy(res, s);
}

static void read_common(const Config *c, Opt *o) {
    UC(lookup_bool(c, "flu.ids", &o->fluids));
    
    UC(lookup_string(c, "dump.strt_base_dump", o->strt_base_dump));
    UC(lookup_string(c, "dump.strt_base_read", o->strt_base_read));

    UC(lookup_bool(c, "dump.field", &o->dump_field));
    UC(conf_lookup_float(c, "dump.freq_field", &o->freq_field));
    
    UC(lookup_bool(c, "dump.strt", &o->dump_strt));
    UC(conf_lookup_float(c, "dump.freq_strt", &o->freq_strt));

    UC(lookup_bool(c, "dump.parts", &o->dump_parts));
    UC(conf_lookup_float(c, "dump.freq_parts", &o->freq_parts));

    UC(conf_lookup_int(c, "sampler.freq", &o->sampler_freq));
    UC(conf_lookup_int3(c, "sampler.grid_ref", &o->sampler_grid_ref));
}

static void set_eq(Opt *o) {
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

static void read_full(const Config *c, Opt *o) {
    UC(lookup_bool(c, "fsi.active", &o->fsi));
    UC(lookup_bool(c, "cnt.active", &o->cnt));

    UC(lookup_bool(c, "flu.colors", &o->flucolors));
    UC(lookup_bool(c, "flu.ids", &o->fluids));
    UC(lookup_bool(c, "flu.stresses", &o->fluss));

    UC(lookup_bool(c, "rbc.active", &o->rbc));
    UC(lookup_bool(c, "rbc.ids", &o->rbcids));
    UC(lookup_bool(c, "rbc.stretch", &o->rbcstretch));
    o->rbcshifttype = get_shifttype(c, "rbc.shifttype");

    UC(lookup_bool(c, "rig.active", &o->rig));
    UC(lookup_bool(c, "rig.bounce", &o->rig_bounce));
    UC(lookup_bool(c, "rig.empty_pp", &o->rig_empty_pp));
    o->rigshifttype = get_shifttype(c, "rig.shifttype");

    UC(lookup_bool(c, "wall.active", &o->wall));    
    UC(lookup_bool(c, "outflow.active", &o->outflow));
    UC(lookup_bool(c, "inflow.active", &o->inflow));
    UC(lookup_bool(c, "denoutflow.active", &o->denoutflow));
    UC(lookup_bool(c, "vcon.active", &o->vcon));

    UC(lookup_bool(c, "dump.rbc_com", &o->dump_rbc_com));
    UC(conf_lookup_float(c, "dump.freq_rbc_com", &o->freq_rbc_com));
    UC(lookup_bool(c, "dump.forces", &o->dump_forces));
    
    UC(conf_lookup_int(c, "flu.recolor_freq", &o->recolor_freq));

    UC(lookup_bool(c, "flu.push", &o->push_flu));
    UC(lookup_bool(c, "rbc.push", &o->push_rbc));
    UC(lookup_bool(c, "rig.push", &o->push_rig));
}

void opt_read_gen(const Config *c, Opt *o) {
    UC(read_common(c, o));
    UC(set_eq(o));
}

void opt_read_full(const Config *c, Opt *o) {
    UC(read_common(c, o));
    UC(read_full(c, o));
}

void opt_check(const Opt *o) {
    if (o->dump_rbc_com && !o->rbcids)
        ERR("Need rbc.ids activated to dump rbc com!");
}
