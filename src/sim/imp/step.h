void step0(float driving_force0, bool wall0, int it) {
    if (solids0) distr_solid();
    // if (rbcs)    distr_rbc();
    if (rbcs)    distribute_rbc();
    forces(wall0);
    dump_diag0(it);
    if (wall0 || solids0) dump_diag_after(it);
    body_force(driving_force0);
    update_solvent();
    if (solids0) update_solid();
    if (rbcs)    update_rbc(it);
    if (wall0) bounce();
    if (sbounce_back && solids0) bounce_solid(it);
}

void step(float driving_force0, bool wall0, int it) {
    distribute_flu();
    step0(driving_force0, wall0, it);
}
