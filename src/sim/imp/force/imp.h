void forces(bool wall0) {
    clear_forces(flu.ff, flu.q.n);
    if (solids0) clear_forces(rig.ff, rig.q.n);
    if (rbcs)    clear_forces(rbc.ff, rbc.q.n);

    forces_dpd(&flu);
    if (wall0 && w::q.n) forces_wall();
    if (rbcs) forces_rbc(&rbc);

    forces_objects(&flu, &rbc, &rig);
    
    dSync();
}
