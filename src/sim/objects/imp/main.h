static void save_mesh_mbr(Mbr *m) {
    if (m->bbdata)
        convert_pp2rr_previous(m->q.n, m->q.pp, m->bbdata->rr_cp);
}

static void save_mesh_rig(Rig *r) {
    if (r->bbdata)
        convert_pp2rr_previous(r->q.ns * r->q.nv, r->q.i_pp, r->bbdata->rr_cp);
}

void objects_save_mesh(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(save_mesh_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(save_mesh_rig(obj->rig[i]));    
}

static void clear_vel_mbr(Mbr *m) {
    UC(scheme_move_clear_vel(m->q.n, m->q.pp));
}

static void clear_vel_rig(Rig *r) {
    UC(scheme_move_clear_vel(r->q.n, r->q.pp));
}

void objects_clear_vel(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(clear_vel_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(clear_vel_rig(obj->rig[i]));    
}

static void advance_rig(float dt, Rig *r) {
    if (!r->q.n) return;
    RigQuants *q = &r->q;
    UC(rig_update(r->pininfo, dt, q->n, r->ff, q->rr0, q->ns, /**/ q->pp, q->ss));
    UC(rig_update_mesh(dt, q->ns, q->ss, q->nv, q->dvv, /**/ q->i_pp));
}

static void ini_mbr_fast_forces(Mbr *m) {
    aD2D(m->ff_fast, m->ff, m->q.n);
}

static void sub_advance_mbr(float dt, const OptMbr *opt, Mbr *m) {
    UC(ini_mbr_fast_forces(m));
    UC(internal_forces_mbr(dt, opt, m));
    UC(scheme_move_apply(dt, m->mass, m->q.n, m->ff_fast, m->q.pp));
}

static void advance_mbr(float dt, const OptMbr *opt, Mbr *m) {
    int i, n = opt->substeps;
    float sub_dt = dt / n;
    for (i = 0; i < n; ++i)
        UC(sub_advance_mbr(sub_dt, opt, m));
}

void objects_advance(float dt, Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(advance_mbr(dt, &obj->opt.mbr[i], obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(advance_rig(dt, obj->rig[i]));
}

static void distribute_mbr(Mbr *m) {
    RbcQuants *q = &m->q;
    MbrDistr  *d = &m->d;

    UC(drbc_build_map(q->nc, q->nv, q->pp, /**/ d->p));
    UC(drbc_pack(q, /**/ d->p));
    UC(drbc_download(/**/d->p));

    UC(drbc_post_send(d->p, d->c));
    UC(drbc_post_recv(d->c, d->u));

    UC(drbc_unpack_bulk(d->p, /**/ q));

    UC(drbc_wait_send(d->c));
    UC(drbc_wait_recv(d->c, d->u));

    UC(drbc_unpack_halo(d->u, /**/ q));
    dSync();
}

static void distribute_rig(Rig *r) {
    RigQuants *q = &r->q;
    RigDistr  *d = &r->d;
    int nv = q->nv;
    
    UC(drig_build_map(q->ns, q->ss, /**/ d->p));
    UC(drig_pack(q->ns, nv, q->ss, q->i_pp, /**/ d->p));
    UC(drig_download(/**/d->p));

    UC(drig_post_send(d->p, d->c));
    UC(drig_post_recv(d->c, d->u));

    UC(drig_unpack_bulk(d->p, /**/ q));

    UC(drig_wait_send(d->c));
    UC(drig_wait_recv(d->c, d->u));

    UC(drig_unpack_halo(d->u, /**/ q));

    q->n = q->ns * q->nps;
    UC(rig_generate(q->ns, q->ss, q->nps, q->rr0, /**/ q->pp));
    dSync();
}

void objects_distribute (Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(distribute_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(distribute_rig(obj->rig[i]));
}

static void update_dpd_prms(float dt, float kBT, Obj *o) {
    if (o->fsi) UC(pair_compute_dpd_sigma(kBT, dt, /**/ o->fsi));
}

void objects_update_dpd_prms(float dt, float kBT, Objects *obj) {
    int i;
    for (i = 0; i < obj->nmbr; ++i) update_dpd_prms(dt, kBT, obj->mbr[i]);
    for (i = 0; i < obj->nrig; ++i) update_dpd_prms(dt, kBT, obj->rig[i]);
}

bool objects_have_bounce(const Objects *o) {
    return o->bb != NULL;
}

static void get_mbr(Mbr *m, PFarrays *pf) {
    PaArray p;
    FoArray f;
    parray_push_pp(m->q.pp, &p);
    farray_push_ff(m->ff, &f);
    UC(pfarrays_push(pf, m->q.n, p, f));
}

static void get_rig(Rig *r, PFarrays *pf) {
    PaArray p;
    FoArray f;
    parray_push_pp(r->q.pp, &p);
    farray_push_ff(r->ff, &f);
    UC(pfarrays_push(pf, r->q.n, p, f));
}

void objects_get_particles_all(Objects *obj, PFarrays *pf) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) get_mbr(obj->mbr[i], pf);
    for (i = 0; i < obj->nrig; ++i) get_rig(obj->rig[i], pf);
}

void objects_get_particles_mbr(Objects *obj, PFarrays *pf) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) get_mbr(obj->mbr[i], pf);
}

static void get_mbr_accel(const Mbr *m, TimeStepAccel *aa) {
    if(m->q.n) UC(time_step_accel_push(aa, m->mass, m->q.n, m->ff));
}

static void get_rig_accel(const Rig *r, TimeStepAccel *aa) {
    if (r->q.n) UC(time_step_accel_push(aa, r->mass, r->q.n, r->ff));
}

void objects_get_accel(const Objects *obj, TimeStepAccel *aa) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) get_mbr_accel(obj->mbr[i], aa);
    for (i = 0; i < obj->nrig; ++i) get_rig_accel(obj->rig[i], aa);    
}

static void get_params_fsi(const Obj *o, const PairParams **prm) {*prm = o->fsi;}

void objects_get_params_fsi(const Objects *obj, const PairParams *prms[]) {
    int i, j = 0;
    for (i = 0; i < obj->nmbr; ++i) get_params_fsi(obj->mbr[i], &prms[j++]);
    for (i = 0; i < obj->nrig; ++i) get_params_fsi(obj->rig[i], &prms[j++]);
}

static void get_params_adhesion(const Obj *o, const PairParams **prm) {*prm = o->adhesion;}

void objects_get_params_adhesion(const Objects *obj, const PairParams *prms[]) {
    int i, j = 0;
    for (i = 0; i < obj->nmbr; ++i) get_params_adhesion(obj->mbr[i], &prms[j++]);
    for (i = 0; i < obj->nrig; ++i) get_params_adhesion(obj->rig[i], &prms[j++]);
}

static void get_params_repulsion(const Obj *o, const WallRepulsePrm **prm) {*prm = o->wall_rep_prm;}

void objects_get_params_repulsion(const Objects *obj, const WallRepulsePrm *prms[]) {
    int i, j = 0;
    for (i = 0; i < obj->nmbr; ++i) get_params_repulsion(obj->mbr[i], &prms[j++]);
    for (i = 0; i < obj->nrig; ++i) get_params_repulsion(obj->rig[i], &prms[j++]);
}

static void restart_mbr(MPI_Comm cart, const char *base, Mbr *m) {
    UC(rbc_strt_quants(cart, m->mesh, base, m->name, RESTART_BEGIN, &m->q));
}

static void restart_rig(MPI_Comm cart, const char *base, Rig *r) {
    UC(rig_strt_quants(cart, r->mesh, base, r->name, RESTART_BEGIN, &r->q));
}

void objects_restart(Objects *o) {
    int i;
    const char *base = o->opt.dump.strt_base_read;
    for (i = 0; i < o->nmbr; ++i) restart_mbr(o->cart, base, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) restart_rig(o->cart, base, o->rig[i]);
    o->active = true;
}

static double local_vol_mbr(const Mbr *m) {
    long nc;
    double V0;
    nc = m->q.nc;
    V0 = rbc_params_get_tot_volume(m->params);
    return nc * V0;
}

double objects_mbr_tot_volume(const Objects *o) {
    int i;
    double loc, tot;
    if (!o->active) return 0;

    loc = 0;

    for (i = 0; i < o->nmbr; ++i)
        loc += local_vol_mbr(o->mbr[i]);
    
    tot = 0;
    MC(m::Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, o->cart));
    
    return tot;
}

static void activate_mbr_bounce(Mbr *m) {
    if (m->bbdata)
        m->active_bounce = true;
}

void objects_activate_mbr_bounce(Objects *o) {
    int i;
    for (i = 0; i < o->nmbr; ++i) activate_mbr_bounce(o->mbr[i]);
}
