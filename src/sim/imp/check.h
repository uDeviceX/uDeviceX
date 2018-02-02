static void check_size(long n, long nmax) {
    if (n < 0 || n > nmax)
        ERR("wrong size: %ld / %ld", n, nmax);
}

static void check_sizes(Sim *s) {
    if (rbcs) UC(check_size(s->rbc.q.nc, MAX_CELL_NUM));
    UC(check_size(s->flu.q.n , s->flu.q.maxp)); 
}

static void check_pos_soft(Sim *s) {
    UC(dbg_check_pos_soft(s->coords, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
    // if (s->solids0) UC(dbg_check_pos_soft(s->coords, "rig",s->dbg, s->rig.q.n, s->rig.q.pp));
    // if (rbcs)       UC(dbg_check_pos_soft(s->coords, "rbc",s->dbg, s->rbc.q.n, s->rbc.q.pp));
}

static void check_vel(float dt0, Sim *s) {
    const Coords *c = s->coords;
    UC(dbg_check_vel(dt0, c, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
    if (s->solids0) UC(dbg_check_vel(dt0, c, "rig", s->dbg, s->rig.q.n, s->rig.q.pp));
    if (rbcs)       UC(dbg_check_vel(dt0, c, "rbc", s->dbg, s->rbc.q.n, s->rbc.q.pp));
}

static void check_forces(float dt0, Sim *s) {
    const Coords *c = s->coords;
    UC(dbg_check_forces(dt0, c, "flu.ff", s->dbg, s->flu.q.n, s->flu.q.pp, s->flu.ff));
    if (s->solids0) UC(dbg_check_forces(dt0, c, "rig.ff", s->dbg, s->rig.q.n, s->rig.q.pp, s->rig.ff));
    if (rbcs)       UC(dbg_check_forces(dt0, c, "rbc.ff", s->dbg, s->rbc.q.n, s->rbc.q.pp, s->rbc.ff));
}
