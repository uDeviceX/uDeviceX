static void check_size(long n, long nmax) {
    if (n < 0 || n > nmax)
        ERR("wrong size: %ld / %ld", n, nmax);
}

static void check_size_mbr(const Mbr *m) {
    UC(check_size(m->q.nc, MAX_CELL_NUM));
}

static void check_size_rig(const Rig *r) {
    UC(check_size(r->q.ns, MAX_SOLIDS));
}

void objects_check_size(const Objects *obj) {
    if (obj->mbr) UC(check_size_mbr(obj->mbr));
    if (obj->rig) UC(check_size_rig(obj->rig));
}

static void check_vel_mbr(const Dbg *dbg, float dt, const Coords *coords, const Mbr *m) {
    UC(dbg_check_vel(dt, coords, "rbc", dbg, m->q.n, m->q.pp));
}

static void check_vel_rig(const Dbg *dbg, float dt, const Coords *coords, const Rig *r) {
    UC(dbg_check_vel(dt, coords, "rig", dbg, r->q.n, r->q.pp));
}

void objects_check_vel(const Objects *o, const Dbg *dbg, float dt) {
    if (o->mbr) UC(check_vel_mbr(dbg, dt, o->coords, o->mbr));
    if (o->rig) UC(check_vel_rig(dbg, dt, o->coords, o->rig));
}

static void check_forces_mbr(const Dbg *dbg, float dt, const Coords *coords, const Mbr *m) {
    UC(dbg_check_forces(dt, coords, "rbc.ff", dbg, m->q.n, m->q.pp, m->ff));
}

static void check_forces_rig(const Dbg *dbg, float dt, const Coords *coords, const Rig *r) {
    UC(dbg_check_forces(dt, coords, "rig.ff", dbg, r->q.n, r->q.pp, r->ff));
}

void objects_check_forces(const Objects *o, const Dbg *dbg, float dt) {
    if (o->mbr) UC(check_forces_mbr(dbg, dt, o->coords, o->mbr));
    if (o->rig) UC(check_forces_rig(dbg, dt, o->coords, o->rig));
}
