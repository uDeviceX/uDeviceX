#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"
#include "constants.cuh"
#include "types.cuh"
#include <mpi.h>
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
void emalloc(size_t, void **);
void efree(void *);
namespace mpicheck { void check(int, const char *, int); }
#define MC(ans) do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)
namespace m { int Comm_rank(MPI_Comm, int *); }
#ifndef UDX_SIM_OBJECTS_PARRAY_COMMON
#define UDX_SIM_OBJECTS_PARRAY_COMMON
struct PaArray {
    bool colors;
    const float *pp;
    const int *cc;
};

struct PaArray_v;
struct PaCArray_v;

struct Particle;

void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);

bool parray_is_colored(const PaArray *a);

void parray_get_view(const PaArray *a, PaArray_v *v);
void parray_get_view(const PaArray *a, PaCArray_v *v);
#endif
#ifndef UDX_SIM_OBJECTS_FARRAY_COMMON
#define UDX_SIM_OBJECTS_FARRAY_COMMON
struct FoArray {
    bool stress;
    float *ff;
    float *ss;
};

struct FoArray_v;
struct FoSArray_v;

struct Force;

void farray_push_ff(Force *ff, FoArray *a);
void farray_push_ss(float *ss, FoArray *a);

void farray_clear(FoArray *);

bool farray_has_stress(const FoArray *a);

void farray_get_view(const FoArray *a, FoArray_v *v);
void farray_get_view(const FoArray *a, FoSArray_v *v);
#endif
#ifndef UDX_SIM_OBJECTS_PFARRAYS_COMMON
#define UDX_SIM_OBJECTS_PFARRAYS_COMMON
struct PFarray {
    PaArray p;
    FoArray f;
    long n;
};

struct PFarrays;

struct PaArray;
struct FoArray;

void pfarrays_ini(PFarrays**);
void pfarrays_fin(PFarrays*);

void pfarrays_clear(PFarrays*);
void pfarrays_push(PFarrays*, long n, PaArray, FoArray);
void pfarrays_push(PFarrays*, PFarray);
int  pfarrays_size(const PFarrays*);
void pfarrays_get(int i, const PFarrays*, long *n, PaArray*, FoArray*);
void pfarrays_get(int i, const PFarrays*, PFarray*);
#endif

struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;
struct PairAdhesion;

void pair_ini(PairParams **);
void pair_fin(PairParams *);

void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p);
void pair_set_lj(float sigma, float eps, PairParams *p);
void pair_set_adhesion(float k1, float k2, PairParams *p);

void pair_set_conf(const Config *, const char *name_space, PairParams *);

void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p);

void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
void pair_get_view_adhesion(const PairParams*, PairAdhesion*);

struct Contact;
struct int3;
struct PairParams;

void cnt_ini(int maxp, int rank, int3 L, int nobj,      Contact **c);
void cnt_fin(Contact *c);

void cnt_build_cells(int nw, const PairParams **prms, const PaWrap *pw,      Contact *c);
void cnt_bulk(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw);
void cnt_halo(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw, Pap26 all_pp, Fop26 all_ff, const int *all_counts);
struct RNDunif;
struct Fsi;
struct int3;

void fsi_ini(int rank, int3 L,      Fsi **fsi);
void fsi_fin(Fsi *fsi);

void fsi_bind_solvent(PaArray pa, Force *ff, int n, const int *starts,      Fsi *fsi);
void fsi_bulk(Fsi *fsi, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw);
void fsi_halo(Fsi *fsi, int nw, const PairParams **prms, Pap26 all_pp, Fop26 all_ff, const int *all_counts);

struct EObjPack;
struct EObjComm;
struct EObjUnpack;

struct EObjPackF;
struct EObjCommF;
struct EObjUnpackF;

void eobj_pack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjPack **p);
void eobj_comm_ini(MPI_Comm cart,      EObjComm **c);
void eobj_unpack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjUnpack **u);

void eobj_pack_fin(EObjPack *p);
void eobj_comm_fin(EObjComm *c);
void eobj_unpack_fin(EObjUnpack *u);

void eobj_build_map(int nw, const PaWrap *ww,      EObjPack *p);

void eobj_pack(int nw, const PaWrap *ww,      EObjPack *p);
void eobj_download(int nw, EObjPack *p);

void eobj_post_recv(EObjComm *c, EObjUnpack *u);
void eobj_post_send(EObjPack *p, EObjComm *c);
void eobj_wait_recv(EObjComm *c, EObjUnpack *u);
void eobj_wait_send(EObjComm *c);

int26 eobj_get_counts(EObjUnpack *u);
void  eobj_get_all_counts(int nw, EObjUnpack *u, int *all_counts);
Pap26 eobj_upload_shift(EObjUnpack *u);
Fop26 eobj_reini_ff(const EObjUnpack *u, EObjPackF *pf);


void eobj_packf_ini(int3 L, int maxd, int maxpsolid, EObjPackF **p);
void eobj_commf_ini(MPI_Comm cart,      EObjCommF **c);
void eobj_unpackf_ini(int3 L, int maxd, int maxpsolid, EObjUnpackF **u);

void eobj_packf_fin(EObjPackF *p);
void eobj_commf_fin(EObjCommF *c);
void eobj_unpackf_fin(EObjUnpackF *u);

void eobj_download_ff(EObjPackF *p);

void eobj_post_recv_ff(EObjCommF *c, EObjUnpackF *u);
void eobj_post_send_ff(EObjPackF *p, EObjCommF *c);
void eobj_wait_recv_ff(EObjCommF *c, EObjUnpackF *u);
void eobj_wait_send_ff(EObjCommF *c);

void eobj_unpack_ff(EObjUnpackF *u, const EObjPack *p, int nw,      FoWrap *ww);

#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON

struct OptFlu {
    bool colors, ids, ss, push;
};


struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};


struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};


struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};


struct OptParams {
    int3 L;
    float kBT;
    int numdensity;
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
#endif

struct ObjInter;

struct Config;
struct Opt;
struct PFarray;
struct PFarrays;

void obj_inter_ini(const Config*, const Opt*, MPI_Comm, float dt, int maxp,      ObjInter**);
void obj_inter_fin(ObjInter*);

void obj_inter_forces(ObjInter*, const PairParams **fsi_prms, const PairParams **cnt_prms, PFarray *flu, const int *flu_start, PFarrays *obj);

struct ObjExch {
    EObjPack *p;
    EObjComm *c;
    EObjUnpack *u;

    EObjPackF *pf;
    EObjCommF *cf;
    EObjUnpackF *uf;
};


struct ObjInter {
    ObjExch *e;
    Contact *cnt;
    Fsi     *fsi;
};

static void ini_obj_exch(MPI_Comm comm, int3 L,      ObjExch **oe) {
    ObjExch *e;
    int maxpsolid = MAX_PSOLID_NUM;
    EMALLOC(1, oe);
    e = *oe;

    UC(eobj_pack_ini   (L, MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid,      &e->p));
    UC(eobj_comm_ini   (comm,                                              &e->c));
    UC(eobj_unpack_ini (L, MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid,      &e->u));

    UC(eobj_packf_ini  (L, MAX_OBJ_DENSITY, maxpsolid,      &e->pf));
    UC(eobj_commf_ini  (comm,                               &e->cf));
    UC(eobj_unpackf_ini(L, MAX_OBJ_DENSITY, maxpsolid,      &e->uf));
}

void obj_inter_ini(const Config *cfg, const Opt *opt, MPI_Comm cart, float dt, int maxp,      ObjInter **oi) {
    int rank;
    ObjInter *o;
    int3 L = opt->params.L;
    EMALLOC(1, oi);
    o = *oi;

    MC(m::Comm_rank(cart, &rank));
    UC(ini_obj_exch(cart, L, &o->e));

    o->cnt = NULL;
    o->fsi = NULL;

    int nobj = opt->nmbr + opt->nrig;
    UC(cnt_ini(maxp, rank, L, nobj,      &o->cnt));
    UC(fsi_ini(rank, L,      &o->fsi));
}


static void fin_obj_exch(     ObjExch *e) {
    UC(eobj_pack_fin   (     e->p));
    UC(eobj_comm_fin   (     e->c));
    UC(eobj_unpack_fin (     e->u));

    UC(eobj_packf_fin  (     e->pf));
    UC(eobj_commf_fin  (     e->cf));
    UC(eobj_unpackf_fin(     e->uf));
    EFREE(e);
}

void obj_inter_fin(ObjInter *o) {
    UC(fin_obj_exch(o->e));
    if (o->cnt) UC(cnt_fin(o->cnt));
    if (o->fsi) UC(fsi_fin(o->fsi));
    EFREE(o);
}
static void fill_wrappers(PFarrays *obj, int *nwrappers, PaWrap *pw, FoWrap *fw) {
    long i, nw, n;
    PaArray p;
    FoArray f;

    (*nwrappers) = nw = pfarrays_size(obj);

    if (nw > MAX_OBJ_TYPES)
        ERR("Too many objects in wrappers : %d/%d", nw, MAX_OBJ_TYPES);

    for (i = 0; i < nw; ++i, ++pw, ++fw) {
        UC(pfarrays_get(i, obj, &n, &p, &f));
        pw->n = fw->n = n;
        pw->pp = (const Particle*) p.pp;
        fw->ff = (Force*) f.ff;
    }
}

static void bind_solvent(PFarray *flu, const int *starts, Fsi *fsi) {
    UC(fsi_bind_solvent(flu->p, (Force*) flu->f.ff, flu->n, starts,      fsi));
}

static void forces_cnt(ObjInter *oi, int nw, const PairParams *prms[], PaWrap *pw, FoWrap *fw) {
    cnt_build_cells(  nw, prms, pw,      oi->cnt);
    cnt_bulk(oi->cnt, nw, prms, pw, fw);
}

static void forces_fsi(ObjInter *oi, int nw, const PairParams *prms[], PaWrap *pw, FoWrap *fw) {
    fsi_bulk(oi->fsi, nw, prms, pw, fw);
}

static bool has_fsi_work(int nw, const PairParams **prms) {
    for (int i = 0; i < nw; ++i)
        if (prms[i]) return true;
    return false;
}

static bool has_work(const ObjInter *o, int nw, const PairParams **fsi_prms) {
    return o->cnt || has_fsi_work(nw, fsi_prms);
}

void obj_inter_forces(ObjInter *oi, const PairParams **fsi_prms, const PairParams **cnt_prms, PFarray *flu, const int *flu_start, PFarrays *obj) {
    PaWrap pw[MAX_OBJ_TYPES];
    FoWrap fw[MAX_OBJ_TYPES];
    int nw = 0;
    ObjExch *e = oi->e;
    int all_counts[26 * MAX_OBJ_TYPES] = {0};

    UC(fill_wrappers(obj,      &nw, pw, fw));
    if (!nw) return;
    if (!has_work(oi, nw, fsi_prms)) return;



    UC(eobj_build_map(nw, pw,      e->p));
    UC(eobj_pack(nw, pw,      e->p));
    UC(eobj_download(nw,      e->p));

    UC(eobj_post_send(e->p, e->c));
    UC(eobj_post_recv(e->c, e->u));



    if (oi->fsi) UC(bind_solvent(flu, flu_start, oi->fsi));

    if (oi->fsi) UC(forces_fsi(oi, nw, fsi_prms, pw, fw));
    if (oi->cnt) UC(forces_cnt(oi, nw, cnt_prms, pw, fw));



    UC(eobj_wait_send(e->c));
    UC(eobj_wait_recv(e->c, e->u));

    eobj_get_all_counts(nw, e->u,      all_counts);
    Pap26 hpp = eobj_upload_shift(e->u);
    Fop26 hff = eobj_reini_ff(e->u, e->pf);

    if (oi->fsi) UC(fsi_halo(oi->fsi, nw, fsi_prms,         hpp, hff, all_counts));
    if (oi->cnt) UC(cnt_halo(oi->cnt, nw, cnt_prms, pw, fw, hpp, hff, all_counts));



    UC(eobj_download_ff(e->pf));

    UC(eobj_post_send_ff(e->pf, e->cf));
    UC(eobj_post_recv_ff(e->cf, e->uf));

    UC(eobj_wait_send_ff(e->cf));
    UC(eobj_wait_recv_ff(e->cf, e->uf));

    UC(eobj_unpack_ff(e->uf, e->p, nw,      fw));
}
