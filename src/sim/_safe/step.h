void step0(float driving_force0, bool wall0, int it) {
    OO;
    SS;
    if (solids0) distr_solid();
    if (rbcs)    distr_rbc();
    OO;
    SS;
    forces(wall0);
    OO;
    SS;
    dump_diag0(it);
    if (wall0 || solids0) dump_diag_after(it);
    body_force(driving_force0);
    OO;
    SS;
    update_solvent();
    OO;
    SS;
    if (solids0) update_solid();
    OO;
    SS;
    if (rbcs)    update_rbc();
    OO;
    SS;
    if (wall0) bounce();
    OO;
    SS;
    if (sbounce_back && solids0) bounce_solid(it);
    OO;
    SS;
}

void step(float driving_force0, bool wall0, int it) {
    odstr();
    step0(driving_force0, wall0, it);
}
