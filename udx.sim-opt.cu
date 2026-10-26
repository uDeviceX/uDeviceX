// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/sim/opt/imp.cpp */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

/* begin src/inc/def.h */
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
/* solvent colors */
#define XMACRO_COLOR(_)                         \
    _(BLUE)                                     \
    _(RED)

#define ENUM_COLOR(a) a##_COLOR
#define make_enum_color(a) ENUM_COLOR(a) ,
enum {
    XMACRO_COLOR(make_enum_color)
    N_COLOR
};
#undef make_enum_color

enum {
    /* wall margins */
    XWM = 6,
    YWM = 6,
    ZWM = 6,

    SAFETY_FACTOR_MAXP = 3,

    /* maximum number of particles per solid */
    MAX_PSOLID_NUM = 12000,

    /* maximum number of solids per node */
    MAX_SOLIDS = 100,

    /* maximum number of object types (solid, rbc, ...) */
    MAX_MBR_TYPES = 2,
    MAX_RIG_TYPES = 2,
    MAX_OBJ_TYPES = MAX_MBR_TYPES + MAX_RIG_TYPES,

    /* maximum number density of particles of the objects */
    MAX_OBJ_DENSITY = 30,

    /* maximum number of red blood cells per node */
    MAX_CELL_NUM = 1500,

    /* maximum texture size in bytes */
    MAX_TEXO_SIZE = 100000000,

    /* safety factor for dpd halo interactions */
    HSAFETY_FACTOR = 10,

    /* safety factor for odist fragments */
    ODSTR_FACTOR = 3,
};

/* for the spring forces r = min(lmax*RBC_SPRING_CAP, r) */
#define RBC_SPRING_CAP (0.99)
/* end src/inc/def.h */
#endif

/* begin src/conf/imp.h */
struct Config;
struct float3;
struct int3;

// tag::mem[]
void conf_ini(/**/ Config **c);
void conf_fin(/**/ Config *c);
// end::mem[]

// tag::ini[]
void conf_read(int argc, char **argv, /**/ Config *cfg);
// end::ini[]

// tag::lookup[]
void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);
// end::lookup[]

// tag::ns[]
void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);
// end::ns[]

// tag::opt[]
bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);
// end::opt[]

// tag::set[]
void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);
// end::set[]

// tag::dump[]
void conf_write_exe(const Config *cfg, FILE *stream);     // <1>
void conf_write_history(const Config *cfg, FILE *stream); // <2>
// end::dump[]
/* end src/conf/imp.h */
/* begin src/utils/error.h */
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)

/* udx check */
#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

/* [B]o[p] [c]heck */
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
/* end src/utils/error.h */
/* begin src/utils/imp.h */
// tag::mem[]
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>
// end::mem[]

// tag::macros[]
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))
// end::macros[]

// tag::stdio[]
void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);
// end::stdio[]

// tag::tools[]
bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
// end::tools[]
/* end src/utils/imp.h */

/* begin src/io/mesh/imp.h */
#ifndef UDX_SIM_MESH_IO_COMMON
#define UDX_SIM_MESH_IO_COMMON
struct Coords;
struct Particle;
struct int4;
struct MeshWrite;
struct MeshRead;
struct Vectors;

enum {MESH_WRITE_EDGE, MESH_WRITE_CENTER};

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead*, const char *directory, /**/ MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);
/* end src/io/mesh/imp.h */
#endif

/* begin src/sim/opt/imp.h */
#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON
/* solvent options */
struct OptFlu {
    bool colors, ids, ss, push;
};

/* common parameters for membranes and rigid objects */
struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};

/* membrane options */
struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};

/* rigid options */
struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};

/* global physical parameters */
struct OptParams {
    int3 L;          /* subdomain sizes */
    float kBT;       /* temperature     */
    int numdensity;  /* number density  */
};

struct OptDump {
    bool       parts,      mesh,      field,      strt, forces;
    float freq_parts, freq_mesh, freq_field, freq_strt, freq_diag;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
};

struct Opt {
    OptFlu flu;
    OptMbr mbr[MAX_MBR_TYPES];
    OptRig rig[MAX_RIG_TYPES];
    int nmbr, nrig;
    OptWall wall;
    OptParams params;
    OptDump dump;
    bool inflow, outflow, denoutflow, vcon;
    int recolor_freq;
    int sampler_npdump;
    int3 sampler_grid_ref;
    bool restart;
    bool tracers;
};

struct Config;

void opt_read(const Config*, Opt*);
void opt_check(const Opt*);
long opt_estimate_maxp(const Opt*);
/* end src/sim/opt/imp.h */
#endif

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

static void read_obj_common(const Config *c, bool restart, const char *ns, OptObj *o) {
    o->shifttype = get_shifttype_ns(c, ns, "shifttype");
    UC(lookup_bool_ns(c, ns, "push",   &o->push));
    UC(lookup_bool_ns(c, ns, "bounce", &o->bounce));
    UC(conf_lookup_float_ns(c, ns, "mass", &o->mass));

    UC(lookup_string_ns(c, ns, "templ_file", o->templ_file));

    if (restart) o->ic_file[0] = '\0';
    else         UC(lookup_string_ns(c, ns, "ic_file", o->ic_file));

    strcpy(o->name, ns);
}

static void read_mbr(const Config *c, bool restart, const char *ns, OptMbr *o) {
    UC(read_obj_common(c, restart, ns, o));
    UC(lookup_bool_ns(c, ns, "ids", &o->ids));
    UC(lookup_bool_ns(c, ns, "stretch", &o->stretch));

    UC(lookup_bool(c, "dump.rbc_com", &o->dump_com));
    UC(conf_lookup_int_ns(c, ns, "substeps", &o->substeps));

    if (o->stretch)
        UC(lookup_string_ns(c, ns, "stretch_file", o->stretch_file));
}

static void read_mbr_array(const Config *c, bool restart, int *nmbr, OptMbr *oo) {
    int i, n;
    const char *ss[MAX_MBR_TYPES];
    UC(conf_lookup_vstring(c, "membranes", MAX_MBR_TYPES, &n, ss));
    for (i = 0; i < n; ++i)
        UC(read_mbr(c, restart, ss[i], &oo[i]));
    *nmbr = n;
}

static void read_rig(const Config *c, bool restart, const char *ns, OptRig *o) {
    UC(read_obj_common(c, restart, ns, o));
    UC(lookup_bool_ns(c, ns, "empty_pp", &o->empty_pp));
}

static void read_rig_array(const Config *c, bool restart, int *nrig, OptRig *oo) {
    int i, n;
    const char *ss[MAX_RIG_TYPES];
    UC(conf_lookup_vstring(c, "rigids", MAX_RIG_TYPES, &n, ss));
    for (i = 0; i < n; ++i)
        UC(read_rig(c, restart, ss[i], &oo[i]));
    *nrig = n;
}

static void read_wall(const Config *c, OptWall *o) {
    UC(lookup_bool(c, "wall.active", &o->active));
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

    UC(read_mbr_array(c, o->restart, &o->nmbr, o->mbr));
    UC(read_rig_array(c, o->restart, &o->nrig, o->rig));

    UC(read_wall(c, &o->wall));
}

static void check_mbr(const OptMbr *m) {
    enum {
        MIN_SUBSTEPS = 1,
        MAX_SUBSTEPS = 1000
    };
    if (m->dump_com && !m->ids)
        ERR("%s: Need ids activated to dump com!", m->name);

    if (m->substeps < MIN_SUBSTEPS || m->substeps > MAX_SUBSTEPS)
        ERR("%s: substeps out of range: %d (min: %d, max: %d)",
            m->name, m->substeps, MIN_SUBSTEPS, MAX_SUBSTEPS);
}

void opt_check(const Opt *o) {
    for (int i = 0; i < o->nmbr; ++i)
        check_mbr(&o->mbr[i]);
}

static long maxp_estimate(const OptParams *p) {
    int3 L = p->L;
    int flu, obj, estimate;
    flu = L.x * L.y * L.z * p->numdensity;
    obj = MAX_OBJ_TYPES * MAX_PSOLID_NUM * MAX_SOLIDS;
    estimate = flu > obj ? flu : obj;
    return SAFETY_FACTOR_MAXP * estimate;
}

long opt_estimate_maxp(const Opt *o) {
    return maxp_estimate(&o->params);
}
/* end src/sim/opt/imp.cpp */

#line 1 "src/sim/walls/imp.cpp"
