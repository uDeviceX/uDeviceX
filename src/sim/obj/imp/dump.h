static void dump_mesh_mbr(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Mbr *m) {
    cD2H(pp, m->q.pp, m->q.n);
    UC(mesh_write_particles(m->mesh_write, cart, coords, m->q.nc, pp, id));
}

static void dump_mesh_rig(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Rig *r) {
    cD2H(pp, r->q.pp, r->q.n);
    UC(mesh_write_particles(r->mesh_write, cart, coords, r->q.ns, pp, id));
}

void objects_mesh_dump(Objects *obj) {
    Dump *d = obj->dump;
    if (obj->mbr) UC(dump_mesh_mbr(obj->cart, obj->coords, d->pp, d->id, obj->mbr));
    if (obj->rig) UC(dump_mesh_rig(obj->cart, obj->coords, d->pp, d->id, obj->rig));
    ++ d->id;
}
