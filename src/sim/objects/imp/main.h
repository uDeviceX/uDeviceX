static void clear_vel_mbr(Mbr *m) {
    UC(scheme_move_clear_vel(m->q.n, m->q.pp));
}

static void clear_vel_rig(Rig *r) {
    UC(scheme_move_clear_vel(r->q.n, r->q.pp));
}

void objects_clear_vel(Objects *obj) {
    if (!obj->active) return;
    if (obj->mbr) UC(clear_vel_mbr(obj->mbr));
    if (obj->rig) UC(clear_vel_rig(obj->rig));    
}

static void update_rig(float dt, Rig *r) {
    if (!r->q.n) return;
    RigQuants *q = &r->q;
    UC(rig_update(r->pininfo, dt, q->n, r->ff, q->rr0, q->ns, /**/ q->pp, q->ss));
    UC(rig_update_mesh(dt, q->ns, q->ss, q->nv, q->dvv, /**/ q->i_pp));
}

static void update_mbr(float dt, Mbr *m) {
    UC(scheme_move_apply(dt, m->mass, m->q.n, m->ff, m->q.pp));
}

void objects_update(float dt, Objects *obj) {
    if (!obj->active) return;
    if (obj->mbr) UC(update_mbr(dt, obj->mbr));
    if (obj->rig) UC(update_rig(dt, obj->rig));
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
    if (!obj->active) return;
    if (obj->mbr) UC(distribute_mbr(obj->mbr));
    if (obj->rig) UC(distribute_rig(obj->rig));
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
    if (!obj->active) return;
    if (obj->mbr) get_mbr(obj->mbr, pf);
    if (obj->rig) get_rig(obj->rig, pf);
}

void objects_get_particles_mbr(Objects *obj, PFarrays *pf) {
    if (!obj->active) return;
    if (obj->mbr) get_mbr(obj->mbr, pf);
}

static void get_mbr_accel(const Mbr *m, TimeStepAccel *aa) {
    if(m->q.n) UC(time_step_accel_push(aa, m->mass, m->q.n, m->ff));
}

static void get_rig_accel(const Rig *r, TimeStepAccel *aa) {
    if (r->q.n) UC(time_step_accel_push(aa, r->mass, r->q.n, r->ff));
}

void objects_get_accel(const Objects *obj, TimeStepAccel *aa) {
    if (!obj->active) return;
    if (obj->mbr) get_mbr_accel(obj->mbr, aa);
    if (obj->rig) get_rig_accel(obj->rig, aa);    
}

static void restart_mbr(MPI_Comm cart, const char *base, Mbr *m) {
    UC(rbc_strt_quants(cart, m->mesh, base, RESTART_BEGIN, &m->q));
}

static void restart_rig(MPI_Comm cart, const char *base, Rig *r) {
    UC(rig_strt_quants(cart, r->mesh, base, RESTART_BEGIN, &r->q));
}

void objects_restart(Objects *o) {
    const char *base = o->opt.strt_base_read;
    if (o->mbr) restart_mbr(o->cart, base, o->mbr);
    if (o->rig) restart_rig(o->cart, base, o->rig);
    o->active = true;
}

double objects_mbr_tot_volume(const Objects *o) {
    double loc, tot, V0;
    long nc;
    Mbr *m = o->mbr;
    if (!o->active) return 0;
    if (!m) return 0;
    
    nc = m->q.nc;
    V0 = rbc_params_get_tot_volume(m->params);

    tot = 0;
    loc = nc * V0;
    MC(m::Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, o->cart));
    
    return tot;
}
