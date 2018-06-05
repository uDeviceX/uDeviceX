static void fin_mbr_distr(/**/ MbrDistr *d) {
    UC(drbc_pack_fin(/**/ d->p));
    UC(drbc_comm_fin(/**/ d->c));
    UC(drbc_unpack_fin(/**/ d->u));
}

static void fin_rig_distr(/**/ RigDistr *d) {
    UC(drig_pack_fin(/**/ d->p));
    UC(drig_comm_fin(/**/ d->c));
    UC(drig_unpack_fin(/**/ d->u));
}

static void fin_mesh_exch(/**/ MeshExch *e) {
    UC(emesh_pack_fin(   /**/ e->p));
    UC(emesh_comm_fin(   /**/ e->c));
    UC(emesh_unpack_fin( /**/ e->u));
    EFREE(e);
}

static void fin_mesh_mom_exch(/**/ MeshMomExch *e) {
    UC(emesh_packm_fin(   /**/ e->p));
    UC(emesh_commm_fin(   /**/ e->c));
    UC(emesh_unpackm_fin( /**/ e->u));
    EFREE(e);
}

static void fin_bbdata(/**/ BounceBackData *bb) {
    UC(fin_mesh_mom_exch(/**/ bb->e));
    Dfree(bb->mm);
    EFREE(bb);
}

static void fin_colorer(/**/ Colorer *c) {
    Dfree(c->pp_mesh);
    Dfree(c->lo);
    Dfree(c->hi);
    EFREE(c);
}

static void fin_obj_common(Obj *o) {
    UC(mesh_read_fin(o->mesh));
    UC(mesh_write_fin(o->mesh_write));

    if (o->fsi)      UC(pair_fin(o->fsi));
    if (o->adhesion) UC(pair_fin(o->adhesion));
    if (o->wall_rep_prm)
        UC(wall_repulse_prm_fin(o->wall_rep_prm));
}

static void fin_mbr(Mbr *m) {
    UC(rbc_fin(&m->q));
    UC(rbc_force_fin(m->force));

    UC(fin_mbr_distr(/**/ &m->d));
        
    Dfree(m->ff);
    Dfree(m->ff_fast);
    UC(triangles_fin(m->tri));

    if (m->com)     UC(rbc_com_fin(/**/ m->com));
    if (m->stretch) UC(rbc_stretch_fin(/**/ m->stretch));

    UC(rbc_params_fin(m->params));

    if (m->colorer)   UC(fin_colorer(/**/ m->colorer));
    if (m->mesh_exch) UC(fin_mesh_exch(/**/ m->mesh_exch));

    UC(fin_obj_common(m));
    
    EFREE(m);
}

static void fin_rig(Rig *r) {
    UC(rig_fin(&r->q));
    Dfree(r->ff);

    UC(fin_rig_distr(/**/ &r->d));
    UC(rig_pininfo_fin(r->pininfo));

    if (r->bbdata) UC(fin_bbdata(r->bbdata));

    UC(fin_obj_common(r));

    EFREE(r);
}

static void fin_dump(Dump *d) {
    EFREE(d->pp);
    EFREE(d);
}

void objects_fin(Objects *obj) {
    int i;
    for (i = 0; i < obj->nmbr; ++i) UC(fin_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(fin_rig(obj->rig[i]));
    if (obj->nmbr) EFREE(obj->mbr);
    if (obj->nrig) EFREE(obj->rig);
    if (obj->bb)  UC(meshbb_fin(/**/ obj->bb));
    UC(fin_dump(obj->dump));
    UC(coords_fin(obj->coords));
    MC(m::Comm_free(&obj->cart));
    EFREE(obj);
}
