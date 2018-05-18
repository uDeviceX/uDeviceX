static void reini_colors(PFarray *flu) {
    UC(collision_label_ini(flu->n, BLUE_COLOR, /**/ (int*) flu->p.cc));
}

static int exchange_mesh_mbr(Mbr *m, Particle *pp) {
    int nm, nv, nmhalo;
    const RbcQuants *q = &m->q;
    MeshExch *e = m->mesh_exch;
    nm = q->nc;
    nv = q->nv;
    
    UC(emesh_build_map(nm, nv, q->pp, /**/ e->p));
    UC(emesh_pack(nv, q->pp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));

    if (nm*nv) aD2D(pp, q->pp, nm*nv);

    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    UC(emesh_unpack(nv, e->u, /**/ &nmhalo, pp + nm * nv));
    return nm + nmhalo;    
}

static void recolor_flu_from_mbr(Mbr *m, PFarray *flu) {
    int nm, nv;
    Colorer *c = m->colorer;

    nv = m->q.nv;
    nm = exchange_mesh_mbr(m, c->pp_mesh);

    UC(minmax(c->pp_mesh, nv, nm, /**/ c->lo, c->hi));

    UC(collision_label(NOT_PERIODIC, flu->n, (const Particle*) flu->p.pp, m->tri, nv, nm,
                       c->pp_mesh, c->lo, c->hi, RED_COLOR, /**/ (int*) flu->p.cc));
}

/* TODO 2 steps recoloring */
/* not supported yet */
void objects_recolor_flu(Objects *obj, PFarray *flu) {
    int i;
    if (!obj->active) return;
    if (!obj->nmbr) return;
    if (flu->n == 0) return;
    if (obj->nmbr > 1) ERR("not implemented yet for more than one membrane type");

    UC(reini_colors(flu));
    for (i = 0; i < obj->nmbr; ++i) UC(recolor_flu_from_mbr(obj->mbr[i], flu));
}
