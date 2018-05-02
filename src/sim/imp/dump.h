static void dump_part(Sim *s) {
    const Flu *flu = &s->flu;
    const Opt *opt = &s->opt;
    IoBop *bop = s->dump.bop;
    int id_bop = s->dump.id_bop;
    cD2H(flu->q.pp_hst, flu->q.pp, flu->q.n);
    if (opt->fluids) {
        cD2H(flu->q.ii_hst, flu->q.ii, flu->q.n);
        io_bop_ids(s->cart, flu->q.n, flu->q.ii_hst, "id_solvent", id_bop, bop);
    }
    if (opt->flucolors) {
        cD2H(flu->q.cc_hst, flu->q.cc, flu->q.n);
        io_bop_colors(s->cart, flu->q.n, flu->q.cc_hst, "colors_solvent", id_bop, bop);
    }
    if (opt->fluss) {
        cD2H(flu->ss_hst, flu->ss, 6 * flu->q.n);
        io_bop_stresses(s->cart, flu->q.n, flu->ss_hst, "stress_solvent", id_bop, bop);
    }

    if (opt->dump_forces) {
        cD2H(flu->ff_hst, flu->ff, flu->q.n);
        io_bop_parts_forces(s->cart, s->coords, flu->q.n, flu->q.pp_hst, flu->ff_hst, "solvent", id_bop, bop);
    } else {
        io_bop_parts(s->cart, s->coords, flu->q.n, flu->q.pp_hst, "solvent", id_bop, bop);
    }
    // TODO
    // if (active_rig(s)) {
    //     cD2H(rig->q.pp_hst, rig->q.pp, rig->q.n);
    //     if (opt->dump_forces) {
    //         cD2H(rig->ff_hst, rig->ff, rig->q.n);
    //         io_bop_parts_forces(s->cart, s->coords, rig->q.n, rig->q.pp_hst, rig->ff_hst, "solid", id_bop, bop);
    //     } else {
    //         io_bop_parts(s->cart, s->coords, rig->q.n, rig->q.pp_hst, "solid", id_bop, bop);
    //     }
    // }
    s->dump.id_bop = ++id_bop;
}

static void dump_grid(Sim *s) {
    Dump *d = &s->dump;
    UC(grid_sampler_dump(s->cart, "h5", d->id_field, d->field_sampler.s));
    UC(grid_sampler_reset(d->field_sampler.s));
    d->id_field ++;
}

void dump_diag_after(const TimeLine *time, Sim *s) { /* after wall */
    const Opt *o = &s->opt;
    float t = time_line_get_current(time);
    if (time_line_cross(time, o->freq_parts)) {
        UC(objects_mesh_dump(s->obj));
        UC(objects_diag_dump(t, s->obj));
    }
}

static int download_pp(Sim *s) { /* device to host  data transfer */
    int np = 0;
    Flu *flu = &s->flu;

    if (flu->q.n) {
        cD2H(s->dump.pp + np, flu->q.pp, flu->q.n);    np += flu->q.n;
    }
    return np;
}

static void diag(float time, Sim *s) {
    if (time < 0) ERR("time = %g < 0", time);
    int n;
    n = download_pp(s);
    UC(diag_part_apply(s->dump.diagpart, s->cart, time, n, s->dump.pp));
}

static void dump_strt_templ(Sim *s) { /* template dumps (wall, solid) */
    const Opt *opt = &s->opt;
    const char *base = opt->strt_base_dump;
    if (opt->dump_strt) {
        if (opt->wall)       wall_dump_templ(s->wall, s->cart, base);
        UC(objects_strt_templ(base, s->obj));
    }
}

static void dump_strt0(int id, Sim *s) {
    Flu *flu = &s->flu;
    const Opt *opt = &s->opt;
    const char *base = opt->strt_base_dump;
    UC(flu_strt_dump(s->cart, base, id, &flu->q));
    UC(objects_strt_dump(base, id, s->obj));
    if (opt->vcon)   vcont_strt_dump(s->cart, base, id, s->vcon.vcont);
}

static void dump_strt(Sim *s) {
    dump_strt0(s->dump.id_strt++, s);
}

static void dump_strt_final(Sim *s) {
    dump_strt0(RESTART_FINAL, s);
}

static void dump_diag(TimeLine *time, Sim *s) {
    const Opt *o = &s->opt;
    if (time_line_cross(time, o->freq_parts)) {
        if (o->dump_parts) dump_part(s);
        UC(diag(time_line_get_current(time), s));
    }
    if (o->dump_field && time_line_cross(time, o->freq_field))
        dump_grid(s);
    if (o->dump_strt  && time_line_cross(time, o->freq_strt))
        dump_strt(s);
}
