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
    UC(dbg_check_pos_soft(s->coords, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
    // if (s->solids0) UC(dbg_check_pos_soft(s->coords, "rig",s->dbg, s->rig.q.n, s->rig.q.pp));
    // if (rbcs)       UC(dbg_check_pos_soft(s->coords, "rbc",s->dbg, s->rbc.q.n, s->rbc.q.pp));
}

void check_vel(Sim *s) {
    Coords c = s->coords;
    UC(dbg_check_vel(c, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
    if (s->solids0) UC(dbg_check_vel(c, "rig", s->dbg, s->rig.q.n, s->rig.q.pp));
    if (rbcs)       UC(dbg_check_vel(c, "rbc", s->dbg, s->rbc.q.n, s->rbc.q.pp));
}

void check_forces(Sim *s) {
    Coords c = s->coords;
    UC(dbg_check_forces(c, "flu.ff", s->dbg, s->flu.q.n, s->flu.q.pp, s->flu.ff));
    if (s->solids0) UC(dbg_check_forces(c, "rig.ff", s->dbg, s->rig.q.n, s->rig.q.pp, s->rig.ff));
    if (rbcs)       UC(dbg_check_forces(c, "rbc.ff", s->dbg, s->rbc.q.n, s->rbc.q.pp, s->rbc.ff));
}
