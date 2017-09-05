#define O {dSync(); dbg::check_pos_pu(o::q.pp, o::q.n, __FILE__, __LINE__, ""); dSync();}

void step0(float driving_force0, bool wall0, int it) {
    O;
    if (solids0) distr_solid();
    O;
    if (rbcs)    distr_rbc();
    O;
    forces(wall0);
    O;
    dump_diag0(it);
    O;
    if (wall0 || solids0) dump_diag_after(it);
    O;
    body_force(driving_force0);
    O;
    update_solvent();
    O;
    if (solids0) update_solid();
}

void step(float driving_force0, bool wall0, int it) {
    O;
    odstr();
    O;
    step0(driving_force0, wall0, it);
}
