static void check_size(long n, long max) {
    if (n < 0 || n >= max)
        ERR("wrong size: %ld / %ld", n, max);
}

void step(scheme::force::Param *fpar, bool wall0, int it) {
    UC(check_size(rbc.q.nc, MAX_CELL_NUM));
    UC(check_size(rbc.q.n , MAX_PART_NUM));
    UC(check_size(flu.q.n , MAX_PART_NUM));
    
    UC(distribute_flu(&flu));
    if (solids0) UC(distribute_rig(/**/ &rig));
    if (rbcs)    UC(distribute_rbc(/**/ &rbc));

    UC(check_size(rbc.q.nc, MAX_CELL_NUM));
    UC(check_size(rbc.q.n , MAX_PART_NUM));
    UC(check_size(flu.q.n , MAX_PART_NUM));

    forces(wall0);

    dump_diag0(it);
    dump_diag_after(it, wall0, solids0);
    body_force(*fpar);

    restrain(it, /**/ &flu, &rbc);
    update_solvent(it, /**/ &flu);
    if (solids0) update_solid(/**/ &rig);
    if (rbcs)    update_rbc(it, &rbc);

    if (VCON && wall0) {
        sample(it, &flu, /**/ &vcont);
        adjust(it, /**/ &vcont, fpar);
        log(it, &vcont);
    }

    if (wall0) bounce_wall(&wall, /**/ &flu, &rbc);

    if (sbounce_back && solids0) bounce_solid(it, /**/ &bb, &rig);

    recolor_flux(/**/ &flu);
}
