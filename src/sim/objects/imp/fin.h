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

static void fin_mbr(Mbr *m) {
    UC(rbc_fin(&m->q));
    UC(rbc_force_fin(m->force));

    UC(fin_mbr_distr(/**/ &m->d));
        
    Dfree(m->ff);
    UC(triangles_fin(m->tri));

    if (m->com)     UC(rbc_com_fin(/**/ m->com));
    if (m->stretch) UC(rbc_stretch_fin(/**/ m->stretch));

    UC(rbc_params_fin(m->params));
    UC(mesh_read_fin(m->cell));
    UC(mesh_write_fin(m->mesh_write));

    EFREE(m);
}

static void fin_rig(Rig *r) {
    UC(rig_fin(&r->q));
    Dfree(r->ff);
    EFREE(r->ff_hst);
    UC(mesh_read_fin(r->mesh));

    UC(fin_rig_distr(/**/ &r->d));
    UC(mesh_write_fin(r->mesh_write));
    UC(rig_pininfo_fin(r->pininfo));

    EFREE(r);
}

static void fin_dump(Dump *d) {
    EFREE(d->pp);
    EFREE(d);
}

void objects_fin(Objects *obj) {
    if (obj->mbr) UC(fin_mbr(obj->mbr));
    if (obj->rig) UC(fin_rig(obj->rig));
    UC(fin_dump(obj->dump));
    UC(coords_fin(obj->coords));
    MC(m::Comm_free(&obj->cart));
    EFREE(obj);
}
