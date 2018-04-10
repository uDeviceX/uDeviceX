void forces(float dt, TimeLine *time, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    bool fluss;
    Opt *opt = &s->opt;

    NVTX_PUSH("forces");
    
    fluss = opt->fluss && time_line_cross(time, opt->freq_parts);

    UC(clear_forces(flu->q.n, flu->ff));
    if (active_rig(s))  UC(clear_forces  (rig->q.n, rig->ff));
    if (active_rbc(s))  UC(clear_forces  (rbc->q.n, rbc->ff));
    if (fluss)          UC(clear_stresses(flu->q.n, flu->ss));
    
    UC(forces_dpd(fluss, flu));
    if (active_walls(s)) forces_wall(fluss, s);
    if (active_rbc(s))   forces_rbc(dt, opt, rbc);

    UC(forces_objects(s));
    
    dSync();

    NVTX_POP();
}
