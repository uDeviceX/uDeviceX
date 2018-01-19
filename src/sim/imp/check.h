static void check_size(long n, long nmax) {
    if (n < 0 || n > nmax)
        ERR("wrong size: %ld / %ld", n, nmax);
}

void check_sizes(Sim *s) {
    UC(check_size(s->rbc.q.nc, MAX_CELL_NUM));
    UC(check_size(s->rbc.q.n , MAX_CELL_NUM * RBCnv));
    UC(check_size(s->flu.q.n , MAX_PART_NUM)); 
}

void check_pos_soft(Sim *s) {
    UC(dbg_check_pos_soft(s->dbg, s->flu.q.n, s->flu.q.pp));
    // if (s->solids0) UC(dbg_check_pos_soft(s->dbg, s->rig.q.n, s->rig.q.pp));
    // if (rbcs)       UC(dbg_check_pos_soft(s->dbg, s->rbc.q.n, s->rbc.q.pp));
}

void check_vel(Sim *s) {
    UC(dbg_check_vel(s->dbg, s->flu.q.n, s->flu.q.pp));
    if (s->solids0) UC(dbg_check_vel(s->dbg, s->rig.q.n, s->rig.q.pp));
    if (rbcs)       UC(dbg_check_vel(s->dbg, s->rbc.q.n, s->rbc.q.pp));
}

void check_forces(Sim *s) {
    UC(dbg_check_forces(s->dbg, s->flu.q.n, s->flu.ff));
    if (s->solids0) UC(dbg_check_forces(s->dbg, s->rig.q.n, s->rig.ff));
    if (rbcs)       UC(dbg_check_forces(s->dbg, s->rbc.q.n, s->rbc.ff));
}
