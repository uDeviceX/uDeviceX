static void gen_mesh_mbr(Coords *coords, MPI_Comm cart, Mbr *m) {
    MeshRead *mesh = m->mesh;
    UC(rbc_gen_mesh(coords, cart, mesh, m->ic_file, /**/ &m->q));
}

static void gen_mesh_rig(Coords *coords, MPI_Comm cart, Rig *r) {
    MeshRead *mesh = r->mesh;
    UC(rig_gen_mesh(coords, cart, mesh, r->ic_file, /**/ &r->q));
}

void objects_gen_mesh(Objects *o) {
    int i;
    for (i = 0; i < o->nmbr; ++i) gen_mesh_mbr(o->coords, o->cart, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) gen_mesh_rig(o->coords, o->cart, o->rig[i]);
}

template <typename T>
static void remove(T *data, int nv, int *stay, int nc) {
    int c; /* c: cell index */
    for (c = 0; c < nc; c++)
        cA2A(data + nv*c, data + nv * stay[c], nv);
}

static void remove_mbr(const Sdf *sdf, Mbr *m) {
    int nc0, stay[MAX_CELL_NUM];
    RbcQuants *q = &m->q;
    q->nc = sdf_who_stays(sdf, q->n, q->pp, nc0 = q->nc, q->nv, /**/ stay);
    q->n = q->nc * q->nv;
    remove(q->pp, q->nv, stay, q->nc);
    msg_print("%s: %d/%d survived", m->name, q->nc, nc0);
}

static void remove_rig(const Sdf *sdf, Rig *r) {
    int nip, ns0, stay[MAX_SOLIDS];
    RigQuants *q = &r->q;
    nip = q->ns * q->nv;
    q->ns = sdf_who_stays(sdf, nip, q->i_pp, ns0 = q->ns, q->nv, /**/ stay);
    q->n  = q->ns * q->nps;
    remove(q->pp,       q->nps,      stay, q->ns);
    remove(q->pp_hst,   q->nps,      stay, q->ns);

    remove(q->ss,       1,           stay, q->ns);
    remove(q->ss_hst,   1,           stay, q->ns);

    remove(q->i_pp,     q->nv, stay, q->ns);
    remove(q->i_pp_hst, q->nv, stay, q->ns);
    msg_print("%s: %d/%d survived", r->name, q->ns, ns0);
}

void objects_remove_from_wall(const Sdf *sdf, Objects *o) {
    int i;
    if (!sdf) return;
    for (i = 0; i < o->nmbr; ++i) remove_mbr(sdf, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) remove_rig(sdf, o->rig[i]);
}

static void gen_freeze_mbr(MPI_Comm cart, Mbr *m) {
    UC(rbc_gen_freeze(cart, /**/ &m->q));
}

static void gen_freeze_rig(const Coords *coords, MPI_Comm cart, bool empty_pp, int numdensity, PFarray *flu, Rig *r) {
    const MeshRead *mesh = r->mesh;
    int n;
    Particle *pp = (Particle*) flu->p.pp;
    n = flu->n;
    
    UC(rig_gen_freeze(coords, empty_pp, numdensity, r->mass, r->pininfo,
                      cart, mesh, pp, &n, /**/ &r->q));
    
    flu->n = n;
}

void objects_gen_freeze(PFarray *flu, Objects *o) {
    int i;
    const Opt *opt = &o->opt;

    for (i = 0; i < o->nmbr; ++i)
        gen_freeze_mbr(o->cart, o->mbr[i]);

    for (i = 0; i < o->nrig; ++i)
        gen_freeze_rig(o->coords, o->cart, opt->rig[i].empty_pp, opt->params.numdensity, flu, o->rig[i]);

    o->active = true;
}
