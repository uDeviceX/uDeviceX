void forces(bool wall0) {
    clear_forces(flu.ff, flu.q.n);
    if (solids0) clear_forces(rig.ff, rig.q.n);
    if (rbcs)    clear_forces(rbc.ff, rbc.q.n);

    forces_dpd(&flu);
    if (wall0 && wall.q.n) forces_wall(&wall);
    if (rbcs) forces_rbc(&rbc);

    forces_objects(&objinter, &flu, &rbc, &rig);
    
    dSync();
}
