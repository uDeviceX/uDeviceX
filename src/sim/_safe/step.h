void step0(float driving_force0, bool wall0, int it) {
    OP;
    SP;
    if (solids0) distr_solid();
    if (rbcs)    distr_rbc();
    OP;
    SP;
    forces(wall0);
    O;
    S;
    dump_diag0(it);
    if (wall0 || solids0) dump_diag_after(it);
    body_force(driving_force0);
    O;
    S;
    update_solvent();
    O;
    S;
    if (solids0) update_solid();
    O;
    S;
    if (rbcs)    update_rbc();
    O;
    S;
    if (wall0) bounce();
    O;
    S;
    if (sbounce_back && solids0) bounce_solid(it);
    O;
    S;
}

void step(float driving_force0, bool wall0, int it) {
    odstr();
    step0(driving_force0, wall0, it);
}
