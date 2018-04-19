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

static void ini_mbr(const Config *cfg, const OptMbr *opt, MPI_Comm cart, int3 L, /**/ Mbr **membrane) {
    int nv;
    const char *directory = "r";
    Mbr *m;
    EMALLOC(1, membrane);
    m = *membrane;

    UC(mesh_read_ini_off("rbc.off", &m->cell));
    UC(mesh_write_ini_from_mesh(cart, opt->shifttype, m->cell, directory, /**/ &m->mesh_write));

    nv = mesh_read_get_nv(m->cell);
    
    Dalloc(&m->ff, MAX_CELL_NUM * nv);
    UC(triangles_ini(m->cell, /**/ &m->tri));
    UC(rbc_ini(opt->ids, m->cell, &m->q));
    UC(ini_mbr_distr(opt->ids, nv, cart, L, /**/ &m->d));

    if (opt->dump_com) UC(rbc_com_ini(nv, MAX_CELL_NUM, /**/ &m->com));
    else m->com = NULL;

    if (opt->stretch)  UC(rbc_stretch_ini("rbc.stretch", nv, /**/ &m->stretch));
    else m->stretch = NULL;

    UC(rbc_params_ini(&m->params));
    UC(rbc_params_set_conf(cfg, m->params));

    UC(rbc_force_ini(m->cell, /**/ &m->force));
    UC(rbc_force_set_conf(m->cell, cfg, m->force));

    UC(conf_lookup_float(cfg, "rbc.mass", &m->mass));
}

static void ini_rig(const Config *cfg, const OptRig *opt, MPI_Comm cart, int maxp, int3 L, /**/ Rig **rigid) {
    const int4 *tt;
    int nv, nt;
    Rig *r;
    EMALLOC(1, rigid);
    r = *rigid;
    
    UC(rig_ini(maxp, &r->q));
    tt = r->q.htt; nv = r->q.nv; nt = r->q.nt;
    UC(mesh_write_ini(cart, opt->shifttype, tt, nv, nt, "s", /**/ &r->mesh_write));

    EMALLOC(maxp, &r->ff_hst);
    Dalloc(&r->ff, maxp);

    UC(ini_rig_distr(r->q.nv, cart, L, /**/ &r->d));

    UC(rig_ini_pininfo(&r->pininfo));
    UC(rig_set_pininfo_conf(cfg, r->pininfo));

    UC(conf_lookup_float(cfg, "rig.mass", &r->mass));
}

void objects_ini(const Config *cfg, const Opt *opt, MPI_Comm cart, int maxp, int3 L, Objects **objects) {
    Objects *obj;
    EMALLOC(1, objects);
    obj = *objects;

    if (opt->rbc.active) UC(ini_mbr(cfg, &opt->rbc, cart,       L, &obj->mbr));  else obj->mbr = NULL;
    if (opt->rig.active) UC(ini_rig(cfg, &opt->rig, cart, maxp, L, &obj->rig));  else obj->rig = NULL;
}
