#define O dbg::check_pos_pu(o::q.pp, o::q.n, __FILE__, __LINE__, "op"), \
        dbg::check_vv(o::q.pp, o::q.n, __FILE__, __LINE__, "ov")
#define F dbg::check_ff(o::ff, o::q.n, __FILE__, __LINE__, "of")

void step0(float driving_force0, bool wall0, int it) {
    if (solids0) distr_solid();
    if (rbcs)    distr_rbc();
    forces(wall0);
    dump_diag0(it);
    if (wall0 || solids0) dump_diag_after(it);
    F;
    body_force(driving_force0);
    O;
    F;
    update_solvent();
    O;
    if (solids0) update_solid();
    if (rbcs)    update_rbc();
    O;
    if (wall0) bounce();
    O;
    if (sbounce_back && solids0) bounce_solid(it);
    O;
}

void step(float driving_force0, bool wall0, int it) {
    odstr();
    step0(driving_force0, wall0, it);
}
