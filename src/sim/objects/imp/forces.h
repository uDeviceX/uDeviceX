static void clear_forces(int n, Force* ff) {
    if (n) DzeroA(ff, n);
}

static void clear_mbr_forces(Mbr *m) {
    UC(clear_forces(m->q.n, m->ff));
}

static void clear_rig_forces(Rig *r) {
    UC(clear_forces(r->q.n, r->ff));
    UC(rig_reinit_ft(r->q.ns, /**/ r->q.ss));
}

void objects_clear_forces(Objects *obj) {
    if (!obj->active) return;
    if (obj->mbr) UC(clear_mbr_forces(obj->mbr));
    if (obj->rig) UC(clear_rig_forces(obj->rig));
}

static void internal_forces_mbr(float dt, const OptMbr *opt, Mbr *m) {
    UC(rbc_force_apply(m->force, m->params, dt, &m->q, /**/ m->ff));
    if (opt->stretch) UC(rbc_stretch_apply(m->q.nc, m->stretch, /**/ m->ff));
}

void objects_internal_forces(float dt, Objects *o) {
    if (!o->active) return;
    if (o->mbr) internal_forces_mbr(dt, &o->opt.rbc, o->mbr);
}

static void bforce_mbr(const Coords *c, const BForce *bf, Mbr *m) {
    UC(bforce_apply(c, m->mass, bf, m->q.n, m->q.pp, /**/ m->ff));
}

static void bforce_rig(const Coords *c, const BForce *bf, Rig *r) {
    UC(bforce_apply(c, r->mass, bf, r->q.n, r->q.pp, /**/ r->ff));
}

void objects_body_forces(const BForce *bf, Objects *o) {
    if (!o->active) return;
    if (o->mbr) bforce_mbr(o->coords, bf, o->mbr);
    if (o->rig) bforce_rig(o->coords, bf, o->rig);
}
