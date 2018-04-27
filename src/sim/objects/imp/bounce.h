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


static void find_and_select_collisions_rig(int3 L, float dt, long nflu, Clist cflu, const Particle *ppflu, const Force *ffflu, int nm, Rig *r, MeshBB *bb) {
    RigQuants *q = &r->q;
    MeshInfo mi;
    mi.nv = q->nv;
    mi.nt = q->nt;
    mi.tt = q->dtt;
    UC(meshbb_find_collisions(dt, nm, mi, q->i_pp, cflu.dims, cflu.starts, cflu.counts, ppflu, ffflu, /**/ bb));
    UC(meshbb_select_collisions(nflu, /**/ bb));    
}

static void mom_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshMomExch *e = bb->e;    
    int counts[NFRAGS];
    
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
    MeshInfo mi;
    mi.nv = q->nv;
    mi.nt = q->nt;
    mi.tt = q->dtt;

    UC(meshbb_collect_rig_momentum(dt, q->ns, mi, q->i_pp, bb->mm, /**/ q->ss));
}

void objects_bounce(float dt, Objects *obj, PFarrays *flu) {
    int nmhalo;
    long nflu = 0; /*TODO*/
    Rig *r = obj->rig;

    if (r) mesh_pack_and_send_rig(r);
    if (r) nmhalo = mesh_recv_unpack_rig(r);

    UC(meshbb_reini(nflu, /**/ obj->bb));
    
    /* perform bounce back */

    if (r) clear_momentum_rig(nmhalo, r);
    
    // UC(meshbb_find_collisions(dt, nm + nmhalo, mi, i_pp, L, ss, cc, pp, flu->ff, /**/ bb->d));
    // UC(meshbb_select_collisions(n, /**/ bb->d));
    // UC(meshbb_bounce(dt, flu->mass, n, bb->d, flu->ff, mi, i_pp, /**/ pp, bb->mm));

    if (r) mom_pack_and_send_rig(r);
    if (r) mom_recv_unpack_rig(r);
    if (r) collect_mom_rig(dt, r);

    // /* for dump */
    // cD2H(qs->ss_dmp_bb, qs->ss, nm);
}
