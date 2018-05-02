void forces(float dt, TimeLine *time, Sim *s) {
    Flu *flu = &s->flu;
    bool fluss, tfluss;
    Opt *opt = &s->opt;

    tfluss = time_line_cross(time, opt->freq_parts) ||
        is_sampling_time(s);
    fluss = opt->fluss && tfluss;

    UC(clear_forces(flu->q.n, flu->ff));
    if (fluss) UC(clear_stresses(flu->q.n, flu->ss));
    UC(objects_clear_forces(s->obj));    
    
    UC(forces_dpd(fluss, flu));
    if (active_walls(s)) UC(forces_wall(fluss, s));
    UC(objects_internal_forces(dt, s->obj));

    UC(forces_objects(s));
    
    dSync();
}
