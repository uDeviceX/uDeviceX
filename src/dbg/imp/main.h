void dbg_ini(Dbg **dbg) {
    Dbg *d;
    int i;
    UC(emalloc(sizeof(Dbg), (void**) dbg));
    d = *dbg;
    for (i = 0; i < DBG_NKIND_; ++i) d->state[i] = 0;
}
void dbg_fin(Dbg *dbg) {
    UC(efree(dbg));
}

void dbg_set(int kind, Dbg *dbg) {
    if (kind < 0 || kind >= DBG_NKIND_)
        ERR("Unrecognised kind of id <%d>", kind);
    dbg->state[kind] = 1;
}


void dbg_check_pos(const Dbg *dbg, int n, const Particle *pp) {

}

void dbg_check_pos_soft(const Dbg *dbg, int n, const Particle *pp) {

}

void dbg_check_vel(const Dbg *dbg, int n, const Particle *pp) {

}

void dbg_check_forces(const Dbg *dbg, int n, const Force *ff) {

}

void dbg_check_colors(const Dbg *dbg, int n, const int *ff) {

}

void dbg_check_clist(const Dbg *dbg, int3 L, const int *starts, const int *counts, int n, const Particle *pp) {

}
