static void ini_mbr_distr(bool ids, int nv, MPI_Comm comm, int3 L, /**/ MbrDistr *d) {
    UC(drbc_pack_ini(ids, L, MAX_CELL_NUM, nv, /**/ &d->p));
    UC(drbc_comm_ini(ids, comm, /**/ &d->c));
    UC(drbc_unpack_ini(ids, L, MAX_CELL_NUM, nv, /**/ &d->u));
}

static void ini_rig_distr(int nv, MPI_Comm comm, int3 L, /**/ RigDistr *d) {
    UC(drig_pack_ini(L, MAX_SOLIDS, nv, /**/ &d->p));
    UC(drig_comm_ini(comm, /**/ &d->c));
    UC(drig_unpack_ini(L,MAX_SOLIDS, nv, /**/ &d->u));
}

static void ini_mbr(const Config *cfg, const Opt *opt, MPI_Comm cart, int3 L, /**/ Mbr *r) {
    int nv;
    const char *directory = "r";
    UC(mesh_read_ini_off("rbc.off", &r->cell));
    UC(mesh_write_ini_from_mesh(cart, opt->rbcshifttype, r->cell, directory, /**/ &r->mesh_write));

    nv = mesh_read_get_nv(r->cell);
    
    Dalloc(&r->ff, MAX_CELL_NUM * nv);
    UC(triangles_ini(r->cell, /**/ &r->tri));
    UC(rbc_ini(opt->rbcids, r->cell, &r->q));
    UC(ini_rbc_distr(opt->rbcids, nv, cart, L, /**/ &r->d));
    if (opt->dump_rbc_com) UC(rbc_com_ini(nv, MAX_CELL_NUM, /**/ &r->com));
    if (opt->rbcstretch)   UC(rbc_stretch_ini("rbc.stretch", nv, /**/ &r->stretch));
    UC(rbc_params_ini(&r->params));
    UC(rbc_params_set_conf(cfg, r->params));

    UC(rbc_force_ini(r->cell, /**/ &r->force));
    UC(rbc_force_set_conf(r->cell, cfg, r->force));

    UC(conf_lookup_float(cfg, "rbc.mass", &r->mass));
}

static void ini_rig(const Config *cfg, MPI_Comm cart, const Opt *opt, int maxp, int3 L, /**/ Rig *s) {
    const int4 *tt;
    int nv, nt;

    UC(rig_ini(maxp, &s->q));
    tt = s->q.htt; nv = s->q.nv; nt = s->q.nt;
    UC(mesh_write_ini(cart, opt->rigshifttype, tt, nv, nt, "s", /**/ &s->mesh_write));

    UC(scan_ini(L.x * L.y * L.z, /**/ &s->ws));
    EMALLOC(maxp, &s->ff_hst);
    Dalloc(&s->ff, maxp);

    UC(ini_rig_distr(s->q.nv, cart, L, /**/ &s->d));

    UC(rig_ini_pininfo(&s->pininfo));
    UC(rig_set_pininfo_conf(cfg, s->pininfo));

    UC(conf_lookup_float(cfg, "rig.mass", &s->mass));
}


void objects_ini() {
    
}
