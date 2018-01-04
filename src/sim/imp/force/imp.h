void forces(bool wall0, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;

    clear_forces(flu->ff, flu->q.n);
    if (solids0) clear_forces(rig->ff, rig->q.n);
    if (rbcs)    clear_forces(rbc->ff, rbc->q.n);

    UC(forces_dpd(flu));
    if (wall0 && wall->q.n) forces_wall(wall, s);
    if (rbcs) forces_rbc(rbc);

    UC(forces_objects(&s->objinter, flu, rbc, rig));
    
    dSync();
}
