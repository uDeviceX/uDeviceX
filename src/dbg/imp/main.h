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
    KL(devdbg::check_pos, (k_cnf(n)), (pp, n));
}

void dbg_check_pos_soft(const Dbg *dbg, int n, const Particle *pp) {
    KL(devdbg::check_pos_pu, (k_cnf(n)), (pp, n));
}

void dbg_check_vel(const Dbg *dbg, int n, const Particle *pp) {
    KL(devdbg::check_vv, (k_cnf(n)), (pp, n));
}

void dbg_check_forces(const Dbg *dbg, int n, const Force *ff) {
    KL(devdbg::check_ff, (k_cnf(n)), (ff, n));
}

void dbg_check_colors(const Dbg *dbg, int n, const int *cc) {
    KL(devdbg::check_cc, (k_cnf(n)), (cc, n));
}

void dbg_check_clist(const Dbg *dbg, int3 L, const int *starts, const int *counts, const Particle *pp) {
    int n = L.x * L.y * L.z;
    KL(devdbg::check_clist, (k_cnf(n)), (L, starts, counts, pp));
}
