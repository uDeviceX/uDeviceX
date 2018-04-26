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
    if (obj->mbr) UC(clear_mbr_forces(obj->mbr));
    if (obj->rig) UC(clear_rig_forces(obj->rig));
}
