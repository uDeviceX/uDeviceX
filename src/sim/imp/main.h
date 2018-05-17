_S_ void gen_flu(Sim *s) {
    Flu *flu = &s->flu;
    UC(flu_gen_quants(s->coords, s->opt.params.numdensity, s->gen_color, &flu->q));
    UC(flu_build_cells(&flu->q));
    if (s->opt.flu.ids) flu_gen_ids(s->cart, flu->q.n, &flu->q);
}

_S_ void gen_wall(Sim *s) {
    Flu *flu = &s->flu;
    Wall *w = s->wall;
    bool dump_sdf = s->opt.dump.field;
    
    if (!s->opt.wall.active) return;

    UC(wall_gen(s->cart, s->coords, s->opt.params, dump_sdf,
                &flu->q.n, flu->q.pp, w));
}

_S_ void freeze(Sim *s) { /* generate */
    Flu *flu = &s->flu;
    const Sdf *sdf = NULL;
    PFarray pfflu;    
    const Opt *opt = &s->opt;
    
    UC(objects_gen_mesh(s->obj));
    dSync();

    UC(gen_wall(s));
    
    if (s->opt.wall.active) UC(wall_get_sdf_ptr(s->wall, &sdf));

    UC(objects_remove_from_wall(sdf, s->obj));
    dSync();
    
    UC(utils_get_pf_flu(s, &pfflu));
    UC(objects_gen_freeze(&pfflu, s->obj));
    UC(utils_set_n_flu_pf(&pfflu, s));
    dSync();
    
    UC(clear_vel(s));
    
    if (opt->flu.colors) {
        Particle *pp = flu->q.pp;
        int n = flu->q.n;
        int *cc = flu->q.cc;
        inter_color_apply_dev(s->coords, s->gen_color, n, pp, /*o*/ cc);
    }
}

void sim_gen(Sim *s) {
    const Opt *opt = &s->opt;
    float tstart = 0;
    s->equilibrating = true;

    UC(gen_flu(s));

    MC(m::Barrier(s->cart));

    run(tstart, s->time.eq, s);

    freeze(/**/ s);
    dSync();

    if (opt->mbr.active && opt->flu.colors) UC(colors_from_rbc(s));

    tstart = s->time.eq;
    pre_run(s);
    run(tstart, s->time.end, s);

    /* final strt dump*/
    if (opt->dump.strt) dump_strt_final(s);
}

_S_ void gen_from_restart(Sim *s) {
    Flu *flu = &s->flu;
    const Opt *opt = &s->opt;
    const char *base = opt->dump.strt_base_read;
    bool dump_sdf = opt->dump.field;
    
    UC(flu_strt_quants(s->cart, base, RESTART_BEGIN, &flu->q));
    UC(objects_restart(s->obj));
    if (opt->wall.active) wall_restart(s->cart, s->coords, opt->params, dump_sdf, base, s->wall);
    if (opt->vcon) vcont_strt_read(base, RESTART_BEGIN, s->vcon.vcont);
}

void sim_strt(Sim *s) {
    const Opt *opt = &s->opt;

    gen_from_restart(s); 

    MC(m::Barrier(s->cart));

    pre_run(s);
    run(s->time.eq, s->time.end, s);
    if (opt->dump.strt) UC(dump_strt_final(s));
}
