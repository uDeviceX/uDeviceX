static MeshInfo mesh_info_rig(const Rig *r) {
    MeshInfo mi;
    mi.nv = r->q.nv;
    mi.nt = r->q.nt;
    mi.tt = r->q.dtt;
    return mi;
}

static void mesh_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    MeshExch  *e = r->mesh_exch;

    UC(emesh_build_map(q->ns, q->nv, q->i_pp, /**/ e->p));
    UC(emesh_pack(q->nv, q->i_pp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));    
}

static int mesh_recv_unpack_rig(Rig *r) {
    MeshExch *e = r->mesh_exch;
    RigQuants *q = &r->q;
    int nmhalo;
    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    /* unpack at the end of current mesh buffer */
    UC(emesh_unpack(q->nv, e->u, /**/ &nmhalo, q->i_pp + q->ns * q->nv));
    return nmhalo;
}

static void clear_momentum(long n, Momentum *mm) {
    if (n) DzeroA(mm, n);
}

static void clear_momentum_rig(int nmhalo, Rig *r) {
    long ntri = r->q.nt * (r->q.ns + nmhalo);
    UC(clear_momentum(ntri, r->bbdata->mm));
}


static void find_and_select_collisions_rig(float dt, long nflu, Clist cflu, const Particle *ppflu, const Force *ffflu, int nm, Rig *r, MeshBB *bb) {
    RigQuants *q = &r->q;
    MeshInfo mi = mesh_info_rig(r);
    UC(meshbb_find_collisions(dt, nm, mi, q->i_pp, cflu.dims, cflu.starts, cflu.counts, ppflu, ffflu, /**/ bb));
    UC(meshbb_select_collisions(nflu, /**/ bb));    
}

static void bounce_rig(float dt, float flu_mass, const MeshBB *bb, long n, const Force *ff, Particle *pp, Rig *r) {
    RigQuants *q = &r->q;
    MeshInfo mi = mesh_info_rig(r);
    UC(meshbb_bounce(dt, flu_mass, n, bb, ff, mi, q->i_pp, /**/ pp, r->bbdata->mm));
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

static void collect_mom_rig(float dt, Rig *r) {
    RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshInfo mi = mesh_info_rig(r);
    
    UC(meshbb_collect_rig_momentum(dt, q->ns, mi, q->i_pp, bb->mm, /**/ q->ss));
}

/* TODO less brutal implementation */
static void bounce_rig(float dt, float flu_mass, const Clist flu_cells, PFarray *flu, MeshBB *bb, Rig *r) {
    int nmhalo;
    Particle *pp = (Particle*) flu->p.pp;
    Force    *ff = (Force*)    flu->f.ff;

    mesh_pack_and_send_rig(r);
    nmhalo = mesh_recv_unpack_rig(r);

    UC(meshbb_reini(flu->n, /**/ bb));
    clear_momentum_rig(nmhalo, r);

    find_and_select_collisions_rig(dt, flu->n, flu_cells, pp, ff, nmhalo + r->q.ns, r, bb);
    
    bounce_rig(dt, flu_mass, bb, flu->n, ff, /**/ pp, r);

    mom_pack_and_send_rig(r);
    mom_recv_unpack_rig(r);
    collect_mom_rig(dt, r);

    /* for dump */
    cD2H(r->q.ss_dmp_bb, r->q.ss, r->q.ns);
}

void objects_bounce(float dt, float flu_mass, const Clist flu_cells, PFarray *flu, Objects *obj) {
    MeshBB *bb = obj->bb;
    int i;
 
    if (!obj->active) return;
    if (!bb) return;
    
    for (i = 0; i < obj->nrig; ++i)
        bounce_rig(dt, flu_mass, flu_cells, flu, bb, obj->rig[i]);
}
