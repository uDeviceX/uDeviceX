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

static void ini_mesh_exch(int3 L, int nv, int max_m, MPI_Comm comm, /**/ MeshExch **me) {
    MeshExch *e;
    EMALLOC(1, me);
    e = *me;
    UC(emesh_pack_ini(L, nv, max_m, /**/ &e->p));
    UC(emesh_comm_ini(comm, /**/ &e->c));
    UC(emesh_unpack_ini(L, nv, max_m, /**/ &e->u));
}

static void ini_mesh_mom_exch(int nt, int max_m, MPI_Comm comm, /**/ MeshMomExch *e) {
    UC(emesh_packm_ini(nt, max_m, /**/ &e->p));
    UC(emesh_commm_ini(comm, /**/ &e->c));
    UC(emesh_unpackm_ini(nt, max_m, /**/ &e->u));
}

static void ini_bbdata(int nt, int max_m, MPI_Comm cart, /**/ BounceBackData **bbdata) {
    BounceBackData *bb;
    EMALLOC(1, bbdata);
    bb = *bbdata;
    UC(ini_mesh_mom_exch(nt, max_m, cart, bb->e));
    Dalloc(&bb->mm, max_m * nt);
}

static void ini_colorer(int nv, int max_m, /**/ Colorer **col) {
    Colorer *c;
    EMALLOC(1, col);
    c = *col;
    Dalloc(&c->pp_mesh, nv * max_m);
    Dalloc(&c->lo, max_m);
    Dalloc(&c->hi, max_m);
}

static void ini_mbr(const Config *cfg, const OptMbr *opt, MPI_Comm cart, int3 L, /**/ Mbr **membrane) {
    int nv, max_m;
    const char *directory = "r";
    Mbr *m;
    EMALLOC(1, membrane);
    m = *membrane;
    
    max_m = MAX_CELL_NUM;
    
    m->com       = NULL;
    m->stretch   = NULL;
    m->colorer   = NULL;
    m->mesh_exch = NULL;
    
    UC(mesh_read_ini_off("rbc.off", &m->mesh));
    UC(mesh_write_ini_from_mesh(cart, opt->shifttype, m->mesh, directory, /**/ &m->mesh_write));

    nv = mesh_read_get_nv(m->mesh);
    
    Dalloc(&m->ff, max_m * nv);
    UC(triangles_ini(m->mesh, /**/ &m->tri));
    UC(rbc_ini(max_m, opt->ids, m->mesh, &m->q));
    UC(ini_mbr_distr(opt->ids, nv, cart, L, /**/ &m->d));

    if (opt->dump_com) UC(rbc_com_ini(nv, max_m, /**/ &m->com));
    if (opt->stretch)  UC(rbc_stretch_ini("rbc.stretch", nv, /**/ &m->stretch));

    UC(rbc_params_ini(&m->params));
    UC(rbc_params_set_conf(cfg, m->params));

    UC(rbc_force_ini(m->mesh, /**/ &m->force));
    UC(rbc_force_set_conf(m->mesh, cfg, m->force));

    UC(conf_lookup_float(cfg, "rbc.mass", &m->mass));

    // if () UC(ini_mesh_exch(L, nv, max_m, cart, /**/ m->mesh_exch));
    // if (opt->flucolors) UC(ini_colorer(nv, max_m, /**/ &m->colorer));
}

static void ini_rig(const Config *cfg, const OptRig *opt, MPI_Comm cart, int maxp, int3 L, /**/ Rig **rigid) {
    Rig *r;
    long max_m = MAX_SOLIDS;
    int nv;
    EMALLOC(1, rigid);
    r = *rigid;

    r->bbdata    = NULL;
    r->mesh_exch = NULL;
    
    UC(mesh_read_ini_ply("rig.ply", &r->mesh));
    UC(mesh_write_ini_from_mesh(cart, opt->shifttype, r->mesh, "s", /**/ &r->mesh_write));
    
    UC(rig_ini(max_m, maxp, r->mesh, &r->q));
    
    EMALLOC(maxp, &r->ff_hst);
    Dalloc(&r->ff, maxp);

    nv = r->q.nv;

    UC(ini_rig_distr(nv, cart, L, /**/ &r->d));

    UC(rig_pininfo_ini(&r->pininfo));
    UC(rig_pininfo_set_conf(cfg, r->pininfo));

    UC(conf_lookup_float(cfg, "rig.mass", &r->mass));

    if (opt->bounce) UC(ini_mesh_exch(L, nv, max_m, cart, /**/ &r->mesh_exch));
    if (opt->bounce) UC(ini_bbdata(r->q.nt, max_m, cart, /**/ &r->bbdata));
}

static void ini_dump(long maxp, Dump **dump) {
    Dump *d;
    EMALLOC(1, dump);
    d = *dump;
    EMALLOC(maxp, &d->pp);
    UC(io_rig_ini(&d->rig));
    d->id = d->id_diag = 0;
}

void objects_ini(const Config *cfg, const Opt *opt, MPI_Comm cart, const Coords *coords, int maxp, Objects **objects) {
    Objects *obj;
    int3 L;
    EMALLOC(1, objects);
    obj = *objects;
    obj->opt = *opt;

    MC(m::Comm_dup(cart, &obj->cart));
    L = subdomain(coords);
    UC(coords_ini(cart, L.x, L.y, L.z, &obj->coords));

    if (opt->rbc.active) UC(ini_mbr(cfg, &opt->rbc, cart,       L, &obj->mbr));  else obj->mbr = NULL;
    if (opt->rig.active) UC(ini_rig(cfg, &opt->rig, cart, maxp, L, &obj->rig));  else obj->rig = NULL;

    if (opt->rig.bounce) UC(meshbb_ini(maxp, /**/ &obj->bb)); else obj->bb = NULL;
        
    UC(ini_dump(maxp, &obj->dump));
}
