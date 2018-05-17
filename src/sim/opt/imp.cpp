#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

#include "inc/def.h"

#include "conf/imp.h"
#include "utils/error.h"
#include "utils/imp.h"

#include "io/mesh/imp.h"

#include "imp.h"

static int str2shifttype(const char *type) {
    if      (same_str(type, "edge"  )) return MESH_WRITE_EDGE;
    else if (same_str(type, "center")) return MESH_WRITE_CENTER;
    else
        ERR("Unrecognised rbc shift type <%s>", type);
    return -1;
}

static int get_shifttype(const Config *c, const char *desc) {
    const char *type;
    UC(conf_lookup_string(c, desc, &type));
    return str2shifttype(type);
 }

static int get_shifttype_ns(const Config *c, const char *ns, const char *d) {
    const char *type;
    UC(conf_lookup_string_ns(c, ns, d, &type));
    return str2shifttype(type);
}

static void lookup_bool(const Config *c, const char *desc, bool *res) {
    int b;
    UC(conf_lookup_bool(c, desc, &b));
    *res = b;
}

static void lookup_bool_ns(const Config *c, const char *ns, const char *desc, bool *res) {
    int b;
    UC(conf_lookup_bool_ns(c, ns, desc, &b));
    *res = b;
}

static void lookup_string(const Config *c, const char *desc, char *res) {
    const char *s;
    UC(conf_lookup_string(c, desc, &s));
    strcpy(res, s);
}

static void lookup_string_ns(const Config *c, const char *ns, const char *desc, char *res) {
    const char *s;
    UC(conf_lookup_string_ns(c, ns, desc, &s));
    strcpy(res, s);
}

static void read_flu(const Config *c, OptFlu *o) {
    UC(lookup_bool(c, "flu.colors", &o->colors));
    UC(lookup_bool(c, "flu.ids", &o->ids));
    UC(lookup_bool(c, "flu.stresses", &o->ss));
    UC(lookup_bool(c, "flu.push", &o->push));
}

static void read_mbr(const Config *c, bool restart, const char *ns, OptMbr *o) {
    UC(lookup_bool_ns(c, ns, "active", &o->active));
    UC(lookup_bool_ns(c, ns, "ids", &o->ids));
    UC(lookup_bool_ns(c, ns, "stretch", &o->stretch));
    o->shifttype = get_shifttype_ns(c, ns, "shifttype");
    UC(lookup_bool_ns(c, ns, "push", &o->push));
    
    UC(lookup_bool(c, "dump.rbc_com", &o->dump_com));
    UC(conf_lookup_float_ns(c, ns, "mass", &o->mass));

    UC(lookup_string_ns(c, ns, "templ_file", o->templ_file));

    if (restart) o->ic_file[0] = '\0';
    else         UC(lookup_string_ns(c, ns, "ic_file", o->ic_file));

    if (o->stretch)
        UC(lookup_string_ns(c, ns, "stretch_file", o->stretch_file));
}

static void read_rig(const Config *c, bool restart, const char *ns, OptRig *o) {
    UC(lookup_bool_ns(c, ns, "active", &o->active));
    UC(lookup_bool_ns(c, ns, "bounce", &o->bounce));
    UC(lookup_bool_ns(c, ns, "empty_pp", &o->empty_pp));
    o->shifttype = get_shifttype_ns(c, ns, "shifttype");
    UC(lookup_bool_ns(c, ns, "push", &o->push));
    UC(conf_lookup_float_ns(c, ns, "mass", &o->mass));

    UC(lookup_string_ns(c, ns, "templ_file", o->templ_file));
    if (restart) o->ic_file[0] = '\0';        
    else         UC(lookup_string_ns(c, ns, "ic_file", o->ic_file));        
}

static void read_wall(const Config *c, OptWall *o) {
    UC(lookup_bool(c, "wall.active", &o->active));
    UC(lookup_bool(c, "wall.repulse", &o->repulse));
}

static void read_params(const Config *c, OptParams *p) {
    UC(conf_lookup_int3 (c, "glb.L",          &p->L         ));
    UC(conf_lookup_float(c, "glb.kBT",        &p->kBT       ));
    UC(conf_lookup_int  (c, "glb.numdensity", &p->numdensity));
}

static void read_dump(const Config *c, OptDump *o) {    
    UC(lookup_string(c, "dump.strt_base_dump", o->strt_base_dump));
    UC(lookup_string(c, "dump.strt_base_read", o->strt_base_read));

    UC(lookup_bool(c, "dump.parts", &o->parts));
    UC(conf_lookup_float(c, "dump.freq_parts", &o->freq_parts));

    UC(lookup_bool(c, "dump.mesh", &o->mesh));
    UC(conf_lookup_float(c, "dump.freq_mesh", &o->freq_mesh));

    UC(lookup_bool(c, "dump.field", &o->field));
    UC(conf_lookup_float(c, "dump.freq_field", &o->freq_field));
    
    UC(lookup_bool(c, "dump.strt", &o->strt));
    UC(conf_lookup_float(c, "dump.freq_strt", &o->freq_strt));

    UC(lookup_bool(c, "dump.forces", &o->forces));
    UC(conf_lookup_float(c, "dump.freq_diag", &o->freq_diag));
}

static void read_common(const Config *c, Opt *o) {
    UC(conf_lookup_int(c, "sampler.n_per_dump", &o->sampler_npdump));
    UC(conf_lookup_int3(c, "sampler.grid_ref", &o->sampler_grid_ref));

    UC(lookup_bool(c, "fsi.active", &o->fsi));
    UC(lookup_bool(c, "cnt.active", &o->cnt));

    UC(lookup_bool(c, "outflow.active", &o->outflow));
    UC(lookup_bool(c, "inflow.active", &o->inflow));
    UC(lookup_bool(c, "denoutflow.active", &o->denoutflow));
    UC(lookup_bool(c, "vcon.active", &o->vcon));
    
    UC(conf_lookup_int(c, "flu.recolor_freq", &o->recolor_freq));

    UC(lookup_bool(c, "glb.restart", &o->restart));
}

void opt_read(const Config *c, Opt *o) {
    UC(read_common(c, o));
    UC(read_params(c, &o->params));
    UC(read_dump(c, &o->dump));
    UC(read_flu(c, &o->flu));

    UC(read_mbr(c, o->restart, "rbc", &o->rbc));
    UC(read_rig(c, o->restart, "rig", &o->rig));

    UC(read_wall(c, &o->wall));
}

void opt_check(const Opt *o) {
    if (o->rbc.dump_com && !o->rbc.ids)
        ERR("Need rbc.ids activated to dump rbc com!");
}

static long maxp_estimate(const OptParams *p) {
    int3 L = p->L;
    int estimate = L.x * L.y * L.z * p->numdensity;
    return SAFETY_FACTOR_MAXP * estimate;
}

long opt_estimate_maxp(const Opt *o) {
    return maxp_estimate(&o->params);
}
