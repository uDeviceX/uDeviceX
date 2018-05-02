_S_ void check_size(long n, long nmax) {
    if (n < 0 || n > nmax)
        ERR("wrong size: %ld / %ld", n, nmax);
}

_I_ void check_sizes(Sim *s) {
    // TODO
    // if (active_rbc(s)) UC(check_size(s->rbc.q.nc, MAX_CELL_NUM));
    UC(check_size(s->flu.q.n , s->flu.q.maxp)); 
}

_I_ void check_pos_soft(Sim *s) {
    UC(dbg_check_pos_soft(s->coords, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
}

_I_ void check_vel(float dt, Sim *s) {
    const Coords *c = s->coords;
    UC(dbg_check_vel(dt, c, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
    // TODO
    // if (active_rig(s))  UC(dbg_check_vel(dt, c, "rig", s->dbg, s->rig.q.n, s->rig.q.pp));
    // if (active_rbc(s))  UC(dbg_check_vel(dt, c, "rbc", s->dbg, s->rbc.q.n, s->rbc.q.pp));
}

_I_ void check_forces(float dt, Sim *s) {
    const Coords *c = s->coords;
    UC(dbg_check_forces(dt, c, "flu.ff", s->dbg, s->flu.q.n, s->flu.q.pp, s->flu.ff));
    // TODO
    // if (active_rig(s)) UC(dbg_check_forces(dt, c, "rig.ff", s->dbg, s->rig.q.n, s->rig.q.pp, s->rig.ff));
    // if (active_rbc(s)) UC(dbg_check_forces(dt, c, "rbc.ff", s->dbg, s->rbc.q.n, s->rbc.q.pp, s->rbc.ff));
}
