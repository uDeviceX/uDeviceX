void forces(float dt, TimeLine *time, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    bool fluss, tfluss;
    Opt *opt = &s->opt;

    tfluss = time_line_cross(time, opt->freq_parts) ||
        is_sampling_time(s);
    fluss = opt->fluss && tfluss;

    UC(clear_forces(flu->q.n, flu->ff));
    if (active_rig(s))  UC(clear_forces  (rig->q.n, rig->ff));
    if (active_rbc(s))  UC(clear_forces  (rbc->q.n, rbc->ff));
    if (fluss)          UC(clear_stresses(flu->q.n, flu->ss));
    
    UC(forces_dpd(fluss, flu));
    if (active_walls(s)) forces_wall(fluss, s);
    if (active_rbc(s))   forces_rbc(dt, opt, rbc);

    UC(forces_objects(s));
    
    dSync();
}
