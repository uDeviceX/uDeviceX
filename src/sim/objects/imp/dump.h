static void dump_mesh_mbr(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Mbr *m) {
    cD2H(pp, m->q.pp, m->q.n);
    UC(mesh_write_particles(m->mesh_write, cart, coords, m->q.nc, pp, id));
}

static void dump_mesh_rig(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Rig *r) {
    cD2H(pp, r->q.i_pp, r->q.ns * r->q.nv);
    UC(mesh_write_particles(r->mesh_write, cart, coords, r->q.ns, pp, id));
}

/* TODO different namings */
void objects_mesh_dump(Objects *obj) {
    int i;
    Dump *d = obj->dump;
    if (!obj->active) return;

    for (i = 0; i < obj->nmbr; ++i)
        UC(dump_mesh_mbr(obj->cart, obj->coords, d->pp, d->id, obj->mbr[i]));

    for (i = 0; i < obj->nrig; ++i)
        UC(dump_mesh_rig(obj->cart, obj->coords, d->pp, d->id, obj->rig[i]));
    ++ d->id;
}

static void dump_diag_mbr(MPI_Comm cart, const Coords *coords, Mbr *m, Dump *d) {
    RbcQuants *q = &m->q;
    long nc = q->nc;
    float3 *rr, *vv;
    if (m->com) {
        UC(rbc_com_apply(m->com, nc, q->pp, /**/ &rr, &vv));
        UC(io_com_dump(cart, coords, d->id_diag, nc, q->ii, rr));
    }
}

static void dump_diag_rig(float t, const Coords *coords, Rig *r, Dump *d) {
    RigQuants *q = &r->q;
    cD2H(q->ss_dmp, q->ss, q->ns);
    UC(io_rig_dump(coords, t, q->ns, q->ss_dmp, q->ss_dmp_bb, d->rig));
}

/* TODO different namings */
void objects_diag_dump(float t, Objects *obj) {
    int i;
    Dump *d = obj->dump;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) dump_diag_mbr(obj->cart, obj->coords, obj->mbr[i], d);
    for (i = 0; i < obj->nrig; ++i) dump_diag_rig(t,         obj->coords, obj->rig[i], d);
    ++ d->id_diag;
}

static void dump_part_rig(MPI_Comm cart, const Coords *coords, long id, Rig *r, IoBop *bop) {
    RigQuants *q = &r->q;
    cD2H(q->pp_hst, q->pp, q->n);
    UC(io_bop_parts(cart, coords, q->n, q->pp_hst, "solid", id, bop));
}

/* TODO different namings */
void objects_part_dump(long id, Objects *o, IoBop *bop) {
    int i;
    if (!o->active) return;    
    for (i = 0; i < o->nrig; ++i) dump_part_rig(o->cart, o->coords, id, o->rig[i], bop);
}

void objects_strt_templ(const char *base, Objects *o) {
    int i;
    if (!o->active) return;
    for (i = 0; i < o->nrig; ++i) UC(rig_strt_dump_templ(o->cart, base, o->rig[i]->name, &o->rig[i]->q));
}

void objects_strt_dump(const char *base, long id, Objects *o) {
    int i;
    if (!o->active) return;
    for (i = 0; i < o->nmbr; ++i) UC(rbc_strt_dump(o->cart, base, o->mbr[i]->name, id, &o->mbr[i]->q));
    for (i = 0; i < o->nrig; ++i) UC(rig_strt_dump(o->cart, base, o->rig[i]->name, id, &o->rig[i]->q));
}
