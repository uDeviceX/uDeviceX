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

static void fin_mbr(const OptMbr *opt, Mbr *m) {
    UC(rbc_fin(&m->q));
    UC(rbc_force_fin(m->force));

    UC(fin_mbr_distr(/**/ &m->d));
        
    Dfree(m->ff);
    UC(triangles_fin(m->tri));

    if (opt->dump_com) UC(rbc_com_fin(/**/ m->com));
    if (opt->stretch)  UC(rbc_stretch_fin(/**/ m->stretch));
    UC(rbc_params_fin(m->params));
    UC(mesh_read_fin(m->cell));
    UC(mesh_write_fin(m->mesh_write));

    EFREE(m);
}

static void fin_rig(Rig *r) {
    UC(rig_fin(&r->q));
    Dfree(r->ff);
    EFREE(r->ff_hst);

    UC(fin_rig_distr(/**/ &r->d));
    UC(mesh_write_fin(r->mesh_write));
    UC(rig_fin_pininfo(r->pininfo));

    EFREE(r);
}

void objects_fin(const Opt *opt, Objects *obj) {
    UC(fin_mbr(&opt->rbc, obj->mbr));
    UC(fin_rig(           obj->rig));
    EFREE(obj);
}
