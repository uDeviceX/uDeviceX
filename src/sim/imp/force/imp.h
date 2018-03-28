void forces(float dt, Time *time, bool wall0, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;
    bool fluss;
    Opt *opt = &s->opt;

    fluss = opt->fluss && time_cross(time, opt->freq_parts);

    UC(clear_forces(flu->ff, flu->q.n));
    if (s->rigids)  UC(clear_forces  (rig->ff, rig->q.n));
    if (opt->rbc)   UC(clear_forces  (rbc->ff, rbc->q.n));
    if (fluss)      UC(clear_stresses(flu->ss, flu->q.n));
    
    UC(forces_dpd(fluss, flu));
    if (wall0 && wall->q.n) forces_wall(fluss, wall, s);
    if (opt->rbc) forces_rbc(dt, opt, rbc);

    UC(forces_objects(s));
    
    dSync();
}
