void forces(bool wall0) {
    clear_forces(o::ff, o::q.n);
    if (solids0) clear_forces(s::ff, s::q.n);
    if (rbcs)    clear_forces(r::ff, r::q.n);

    forces_dpd();
    if (wall0 && w::q.n) forces_wall();
    forces_rbc();

    // forces_objects();
    forces_objects_new();
    
    dSync();
}
