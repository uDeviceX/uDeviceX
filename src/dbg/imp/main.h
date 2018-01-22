void dbg_ini(Dbg **dbg) {
    Dbg *d;
    int i;
    UC(emalloc(sizeof(Dbg), (void**) dbg));
    d = *dbg;
    for (i = 0; i < DBG_NKIND_; ++i) d->state[i] = 0;
    d->verbose = false;
}
void dbg_fin(Dbg *dbg) {
    UC(efree(dbg));
}

static void set(int kind, int val, Dbg *dbg) {
    if (kind < 0 || kind >= DBG_NKIND_)
        ERR("Unrecognised kind of id <%d>", kind);
    dbg->state[kind] = val;
}

void dbg_enable(int kind, Dbg *dbg) {
    UC(set(kind, 1, dbg));
}

void dbg_disable(int kind, Dbg *dbg) {
    UC(set(kind, 0, dbg));
}

void dbg_set_verbose(bool tf, Dbg *dbg) {dbg->verbose = tf;}

static int check(const Dbg *dbg, int kind) {return dbg->state[kind];}

static bool error() {
    err_type e;
    UC(e = err_get());
    return e != ERR_NONE;
} 

static void dump() {
    
}

static void print() {
    err_type e;
    UC(e = err_get());
    ERR("DBG: error: %s", get_err_str(e));
} 

void dbg_check_pos(const Dbg *dbg, int n, const Particle *pp) {
    if (!check(dbg, DBG_POS))
        return;
    UC(err_ini());
    KL(devdbg::check_pos, (k_cnf(n)), (pp, n));
    if (error()) {
        UC(print());
    }
}

void dbg_check_pos_soft(const Dbg *dbg, int n, const Particle *pp) {
    if (!check(dbg, DBG_POS_SOFT))
        return;
    UC(err_ini());
    KL(devdbg::check_pos_pu, (k_cnf(n)), (pp, n));
    if (error()) {
        UC(print());
    }
}

void dbg_check_vel(const Dbg *dbg, int n, const Particle *pp) {
    if (!check(dbg, DBG_VEL))
        return;
    UC(err_ini());
    KL(devdbg::check_vv, (k_cnf(n)), (pp, n));
    if (error()) {
        UC(print());
    }
}

void dbg_check_forces(const Dbg *dbg, int n, const Force *ff) {
    if (!check(dbg, DBG_FORCES))
        return;
    UC(err_ini());
    KL(devdbg::check_ff, (k_cnf(n)), (ff, n));
    if (error()) {
        UC(print());
    }
}

void dbg_check_colors(const Dbg *dbg, int n, const int *cc) {
    if (!check(dbg, DBG_COLORS))
        return;
    UC(err_ini());
    KL(devdbg::check_cc, (k_cnf(n)), (cc, n));
    if (error()) {
        UC(print());
    }
}

void dbg_check_clist(const Dbg *dbg, int3 L, const int *starts, const int *counts, const Particle *pp) {
    if (!check(dbg, DBG_CLIST))
        return;
    int n = L.x * L.y * L.z;
    UC(err_ini());
    KL(devdbg::check_clist, (k_cnf(n)), (L, starts, counts, pp));
    if (error()) {
        UC(print());
    }
}
