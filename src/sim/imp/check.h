_S_ void check_size(long n, long nmax) {
    if (n < 0 || n > nmax)
        ERR("wrong size: %ld / %ld", n, nmax);
}

_I_ void check_sizes(Sim *s) {
    UC(check_size(s->flu.q.n , s->flu.q.maxp));
    UC(objects_check_size(s->obj));
}

_I_ void check_pos_soft(Sim *s) {
    UC(dbg_check_pos_soft(s->coords, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
}

_I_ void check_vel(float dt, Sim *s) {
    const Coords *c = s->coords;
    UC(dbg_check_vel(dt, c, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
    UC(objects_check_vel(s->obj, s->dbg, dt));
}

_I_ void check_forces(float dt, Sim *s) {
    const Coords *c = s->coords;
    UC(dbg_check_forces(dt, c, "flu.ff", s->dbg, s->flu.q.n, s->flu.q.pp, s->flu.ff));
    UC(objects_check_forces(s->obj, s->dbg, dt));
}
