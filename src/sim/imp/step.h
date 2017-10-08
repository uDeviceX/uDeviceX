void step(scheme::Fparams *fpar, bool wall0, int it) {
    distribute_flu();
    if (solids0) distribute_rig();
    if (rbcs)    distribute_rbc();

    forces(wall0);

    dump_diag0(it);
    if (wall0 || solids0) dump_diag_after(it);
    body_force(*fpar);

    restrain(it);
    update_solvent(it);
    if (solids0) update_solid();
    if (rbcs)    update_rbc(it);

    if (wall0 && VCON) {
        sample(it, o::q.n, o::q.pp, o::q.cells.starts, /**/ &o::vcont);
        adjust(it, /**/ &o::vcont, fpar);
        log(it, &o::vcont);
    }
    
    if (wall0) bounce_wall();

    if (sbounce_back && solids0) bounce_solid(it);
}
