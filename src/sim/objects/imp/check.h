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

void objects_check_vel(const Objects*, float dt) {

}

void objects_check_forces(const Objects*, float dt) {

}
