static void ini_obj_exch(MPI_Comm comm, int3 L, /**/ ObjExch **oe) {
    ObjExch *e;
    int maxpsolid = MAX_PSOLID_NUM;
    EMALLOC(1, oe);
    e = *oe;

    UC(eobj_pack_ini   (L, MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->p));
    UC(eobj_comm_ini   (comm,                          /**/ &e->c));
    UC(eobj_unpack_ini (L, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->u));
    UC(eobj_packf_ini  (L, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->pf));
    UC(eobj_unpackf_ini(L, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->uf));
}

// TODO: this is copy/paste from sim
static void set_params(const Config *cfg, float kBT, float dt, const char *name_space, PairParams *p) {
    UC(pair_set_conf(cfg, name_space, p));
    UC(pair_compute_dpd_sigma(kBT, dt, /**/ p));
}

static void ini_pair_params(const Config *cfg, const char *base, float kBT, float dt, PairParams **par) {
    PairParams *p;
    UC(pair_ini(par));
    p = *par;
    UC(set_params(cfg, kBT, dt, base, p));
}

void obj_inter_ini(const Config *cfg, const Opt *opt, MPI_Comm cart, float dt, int maxp, /**/ ObjInter **oi) {
    int rank;
    ObjInter *o;
    int3 L = opt->params.L;
    float kBT = opt->params.kBT;
    EMALLOC(1, oi);
    o = *oi;
    
    MC(m::Comm_rank(cart, &rank));
    UC(ini_obj_exch(cart, L, &o->e));

    o->cnt = NULL;
    o->fsi = NULL;
    
    if (opt->cnt) UC(cnt_ini(maxp, rank, L, /**/ &o->cnt));
    if (opt->fsi) UC(fsi_ini(      rank, L, /**/ &o->fsi));

    o->cntparams = NULL;
    o->fsiparams = NULL;
    
    if (opt->cnt) UC(ini_pair_params(cfg, "cnt", kBT, dt, &o->cntparams));
    if (opt->fsi) UC(ini_pair_params(cfg, "fsi", kBT, dt, &o->fsiparams));                     
}


static void fin_obj_exch(/**/ ObjExch *e) {
    UC(eobj_pack_fin   (/**/ e->p));
    UC(eobj_comm_fin   (/**/ e->c));
    UC(eobj_unpack_fin (/**/ e->u));
    UC(eobj_packf_fin  (/**/ e->pf));
    UC(eobj_unpackf_fin(/**/ e->uf));
    EFREE(e);
}

void obj_inter_fin(ObjInter *o) {
    UC(fin_obj_exch(o->e));
    if (o->cnt) UC(cnt_fin(o->cnt));
    if (o->fsi) UC(fsi_fin(o->fsi));
    if (o->cntparams) UC(pair_fin(o->cntparams));
    if (o->fsiparams) UC(pair_fin(o->fsiparams));
}
