static void dump_mesh_mbr(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Mbr *m) {
    cD2H(pp, m->q.pp, m->q.n);
    UC(mesh_write_particles(m->mesh_write, cart, coords, m->q.nc, pp, id));
}

static void dump_mesh_rig(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Rig *r) {
    cD2H(pp, r->q.i_pp, r->q.ns * r->q.nv);
    UC(mesh_write_particles(r->mesh_write, cart, coords, r->q.ns, pp, id));
}

void objects_mesh_dump(Objects *obj) {
    Dump *d = obj->dump;
    if (!obj->active) return;
    if (obj->mbr) UC(dump_mesh_mbr(obj->cart, obj->coords, d->pp, d->id, obj->mbr));
    if (obj->rig) UC(dump_mesh_rig(obj->cart, obj->coords, d->pp, d->id, obj->rig));
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

void objects_diag_dump(float t, Objects *obj) {
    Dump *d = obj->dump;
    if (!obj->active) return;
    if (obj->mbr) dump_diag_mbr(obj->cart, obj->coords, obj->mbr, d);
    if (obj->rig) dump_diag_rig(t,         obj->coords, obj->rig, d);
    ++ d->id_diag;
}

static void dump_part_rig(MPI_Comm cart, const Coords *coords, long id, Rig *r, IoBop *bop) {
    RigQuants *q = &r->q;
    cD2H(q->pp_hst, q->pp, q->n);
    UC(io_bop_parts(cart, coords, q->n, q->pp_hst, "solid", id, bop));
}

void objects_part_dump(long id, Objects *o, IoBop *bop) {
    if (!o->active) return;    
    if (o->rig) dump_part_rig(o->cart, o->coords, id, o->rig, bop);
}

void objects_strt_templ(const char *base, Objects *o) {
    if (!o->active) return;
    if (o->rig) UC(rig_strt_dump_templ(o->cart, base, &o->rig->q));
}

void objects_strt_dump(const char *base, long id, Objects *o) {
    if (!o->active) return;
    if (o->mbr) UC(rbc_strt_dump(o->cart, base, id, &o->mbr->q));
    if (o->rig) UC(rig_strt_dump(o->cart, base, id, &o->rig->q));
}
