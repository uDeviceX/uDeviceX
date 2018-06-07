static MeshInfo mesh_info_mbr(const Mbr *m) {
    MeshInfo mi;
    mi.nv = m->q.nv;
    mi.nt = m->q.nt;
    mi.tt = m->tri->tt;
    return mi;
}

static MeshInfo mesh_info_rig(const Rig *r) {
    MeshInfo mi;
    mi.nv = r->q.nv;
    mi.nt = r->q.nt;
    mi.tt = r->q.dtt;
    return mi;
}

static void mesh_pack_and_send_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    MeshExch *e = m->mesh_exch;

    UC(emesh_build_map(q->nc, q->nv, q->pp, /**/ e->p));
    UC(emesh_pack_rrcp(q->nv, m->bbdata->rr_cp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));    
}

static void mesh_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    MeshExch  *e = r->mesh_exch;

    UC(emesh_build_map(q->ns, q->nv, q->i_pp, /**/ e->p));
    UC(emesh_pack_rrcp(q->nv, r->bbdata->rr_cp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));    
}

static int mesh_recv_unpack_mbr(Mbr *m) {
    MeshExch *e = m->mesh_exch;
    RbcQuants *q = &m->q;
    int nmhalo;
    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    /* unpack at the end of current mesh buffer */
    UC(emesh_unpack_rrcp(q->nv, e->u, /**/ &nmhalo, m->bbdata->rr_cp + q->n));
    return nmhalo;
}

static int mesh_recv_unpack_rig(Rig *r) {
    MeshExch *e = r->mesh_exch;
    RigQuants *q = &r->q;
    int nmhalo;
    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    /* unpack at the end of current mesh buffer */
    UC(emesh_unpack_rrcp(q->nv, e->u, /**/ &nmhalo, r->bbdata->rr_cp + q->ns * q->nv));
    return nmhalo;
}

static void clear_momentum(long n, Momentum *mm) {
    if (n) DzeroA(mm, n);
}

static void clear_momentum_mbr(int nmhalo, Mbr *m) {
    long ntri = m->q.nt * (m->q.nc + nmhalo);
    UC(clear_momentum(ntri, m->bbdata->mm));
}

static void clear_momentum_rig(int nmhalo, Rig *r) {
    long ntri = r->q.nt * (r->q.ns + nmhalo);
    UC(clear_momentum(ntri, r->bbdata->mm));
}


static void find_and_select_collisions_rig(float dt, long nflu, Clist cflu, const Particle *flu_pp, const Particle *flu_pp0, int nm, Rig *r, MeshBB *bb) {
    MeshInfo mi = mesh_info_rig(r);
    UC(mesh_bounce_find_collisions(dt, nm, mi, r->bbdata->rr_cp, cflu.dims, cflu.starts, cflu.counts, flu_pp, flu_pp0, /**/ bb));
    UC(mesh_bounce_select_collisions(nflu, /**/ bb));    
}

static void find_and_select_collisions_mbr(float dt, long nflu, Clist cflu, const Particle *flu_pp, const Particle *flu_pp0, int nm, Mbr *m, MeshBB *bb) {
    MeshInfo mi = mesh_info_mbr(m);
    UC(mesh_bounce_find_collisions(dt, nm, mi, m->bbdata->rr_cp, cflu.dims, cflu.starts, cflu.counts, flu_pp, flu_pp0, /**/ bb));
    UC(mesh_bounce_select_collisions(nflu, /**/ bb));    
}

static void bounce_mbr(float dt, float flu_mass, const MeshBB *bb, long n, const Particle *pp0, Particle *pp, Mbr *m) {
    MeshInfo mi = mesh_info_mbr(m);
    UC(mesh_bounce_bounce(dt, flu_mass, n, bb, mi, m->bbdata->rr_cp, pp0, /**/ pp, m->bbdata->mm));
}

static void bounce_rig(float dt, float flu_mass, const MeshBB *bb, long n, const Particle *pp0, Particle *pp, Rig *r) {
    MeshInfo mi = mesh_info_rig(r);
    UC(mesh_bounce_bounce(dt, flu_mass, n, bb, mi, r->bbdata->rr_cp, pp0, /**/ pp, r->bbdata->mm));
}

static void mom_pack_and_send_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshMomExch *e = bb->e;    
    int counts[NFRAGS];

    UC(emesh_get_num_frag_mesh(m->mesh_exch->u, /**/ counts));
    UC(emesh_packM(q->nt, counts, bb->mm + q->n, /**/ e->p));
    UC(emesh_downloadM(counts, /**/ e->p));

    UC(emesh_post_recv(e->c, e->u));
    UC(emesh_post_send(e->p, e->c));
}

static void mom_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshMomExch *e = bb->e;    
    int counts[NFRAGS];

    UC(emesh_get_num_frag_mesh(r->mesh_exch->u, /**/ counts));
    UC(emesh_packM(q->nt, counts, bb->mm + q->ns * q->nt, /**/ e->p));
    UC(emesh_downloadM(counts, /**/ e->p));

    UC(emesh_post_recv(e->c, e->u));
    UC(emesh_post_send(e->p, e->c));
}

static void mom_recv_unpack_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshMomExch *e = bb->e;
    MeshExch *meshe = m->mesh_exch;
    
    UC(emesh_wait_recv(e->c, e->u));
    UC(emesh_wait_send(e->c));

    UC(emesh_upload(e->u));
    UC(emesh_unpack_mom(q->nt, meshe->p, e->u, /**/ bb->mm));
}

static void mom_recv_unpack_rig(Rig *r) {
    const RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshMomExch *e = bb->e;
    MeshExch *meshe = r->mesh_exch;
    
    UC(emesh_wait_recv(e->c, e->u));
    UC(emesh_wait_send(e->c));

    UC(emesh_upload(e->u));
    UC(emesh_unpack_mom(q->nt, meshe->p, e->u, /**/ bb->mm));
}

static void collect_mom_mbr(float dt, Mbr *m) {
    RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshInfo mi = mesh_info_mbr(m);

    // TODO where to store forces?
    UC(mesh_bounce_collect_rbc_momentum(dt, q->nc, mi, q->pp, bb->mm, /**/ m->ff));
}

static void collect_mom_rig(Rig *r) {
    RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshInfo mi = mesh_info_rig(r);
    
    UC(mesh_bounce_collect_rig_momentum(q->ns, mi, q->i_pp, bb->mm, /**/ q->ss));
}

/* TODO less brutal implementation */
static void bounce_mbr(float dt, float flu_mass, const Clist flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, MeshBB *bb, Mbr *m) {
    int nmhalo;
    if (!m->bbdata) return;

    mesh_pack_and_send_mbr(m);
    nmhalo = mesh_recv_unpack_mbr(m);

    UC(mesh_bounce_reini(n, /**/ bb));
    clear_momentum_mbr(nmhalo, m);

    find_and_select_collisions_mbr(dt, n, flu_cells, flu_pp, flu_pp0, nmhalo + m->q.nc, m, bb);
    
    bounce_mbr(dt, flu_mass, bb, n, flu_pp0, /**/ flu_pp, m);

    mom_pack_and_send_mbr(m);
    mom_recv_unpack_mbr(m);
    collect_mom_mbr(dt, m);
}

static void bounce_rig(float dt, float flu_mass, const Clist flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, MeshBB *bb, Rig *r) {
    int nmhalo;
    if (!r->bbdata) return;

    mesh_pack_and_send_rig(r);
    nmhalo = mesh_recv_unpack_rig(r);

    UC(mesh_bounce_reini(n, /**/ bb));
    clear_momentum_rig(nmhalo, r);

    find_and_select_collisions_rig(dt, n, flu_cells, flu_pp, flu_pp0, nmhalo + r->q.ns, r, bb);
    
    bounce_rig(dt, flu_mass, bb, n, flu_pp0, /**/ flu_pp, r);

    mom_pack_and_send_rig(r);
    mom_recv_unpack_rig(r);
    collect_mom_rig(r);
}

static void save_mesh_mbr_current(Mbr *m) {
    if (m->bbdata)
        convert_pp2rr_current(m->q.n, m->q.pp, m->bbdata->rr_cp);
}

static void save_mesh_rig_current(Rig *r) {
    if (r->bbdata)
        convert_pp2rr_current(r->q.ns * r->q.nv, r->q.i_pp, r->bbdata->rr_cp);
}

static void objects_save_mesh_current(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(save_mesh_mbr_current(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(save_mesh_rig_current(obj->rig[i]));    
}

void objects_bounce(float dt, float flu_mass, const Clist flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, Objects *obj) {
    MeshBB *bb = obj->bb;
    int i;
 
    if (!obj->active) return;
    if (!bb) return;

    UC(objects_save_mesh_current(obj));
    
    for (i = 0; i < obj->nmbr; ++i)
        bounce_mbr(dt, flu_mass, flu_cells, n, flu_pp0, flu_pp, bb, obj->mbr[i]);

    for (i = 0; i < obj->nrig; ++i)
        bounce_rig(dt, flu_mass, flu_cells, n, flu_pp0, flu_pp, bb, obj->rig[i]);
}
