_S_ long download_all_pp(Sim *s) { /* device to host  data transfer */
    long i, np = 0;
    Flu *flu = &s->flu;
    PFarrays *pf;
    PFarray p;
    UC(pfarrays_ini(&pf));
    UC(objects_get_particles_all(s->obj, pf));
    
    if (flu->q.n) {
        aD2H(s->dump.pp + np, flu->q.pp, flu->q.n);
        np += flu->q.n;
    }
    for (i = 0; i < pfarrays_size(pf); ++i) {
        pfarrays_get(i, pf, &p);
        aD2H(s->dump.pp + np, (const Particle*) p.p.pp, p.n);
        np += p.n;
    }
        
    UC(pfarrays_fin(pf));
    dSync();
    return np;
}

_S_ void dump_diag(float t, Sim *s) {
    if (t < 0) ERR("time = %g < 0", t);
    long n;
    n = download_all_pp(s);
    UC(diag_part_apply(s->dump.diagpart, s->cart, t, n, s->dump.pp));
    UC(objects_diag_dump(t, s->obj));
}

_S_ void dump_parts(Sim *s) {
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

    if (opt->dump.forces) {
        cD2H(flu->ff_hst, flu->ff, flu->q.n);
        io_bop_parts_forces(s->cart, s->coords, flu->q.n, flu->q.pp_hst, flu->ff_hst, "solvent", id_bop, bop);
    } else {
        io_bop_parts(s->cart, s->coords, flu->q.n, flu->q.pp_hst, "solvent", id_bop, bop);
    }

    UC(objects_part_dump(id_bop, s->obj, bop));
    s->dump.id_bop = ++id_bop;
}

_S_ void dump_mesh(Sim *s) {
    UC(objects_mesh_dump(s->obj));
}

_S_ void dump_field(Sim *s) {
    Dump *d = &s->dump;
    UC(grid_sampler_dump(s->cart, "h5", d->id_field, d->field_sampler.s));
    UC(grid_sampler_reset(d->field_sampler.s));
    d->id_field ++;
}

_I_ void dump_strt_templ(Sim *s) { /* template dumps (wall, solid) */
    const Opt *opt = &s->opt;
    const char *base = opt->dump.strt_base_dump;
    if (opt->dump.strt) {
        if (opt->wall) UC(wall_dump_templ(s->wall, s->cart, base));
        UC(objects_strt_templ(base, s->obj));
    }
}

_S_ void dump_strt0(int id, Sim *s) {
    Flu *flu = &s->flu;
    const Opt *opt = &s->opt;
    const char *base = opt->dump.strt_base_dump;
    UC(flu_strt_dump(s->cart, base, id, &flu->q));
    UC(objects_strt_dump(base, id, s->obj));
    if (opt->vcon)   vcont_strt_dump(s->cart, base, id, s->vcon.vcont);
}

_S_ void dump_strt(Sim *s) {
    dump_strt0(s->dump.id_strt++, s);
}

_I_ void dump_strt_final(Sim *s) {
    dump_strt0(RESTART_FINAL, s);
}

_I_ void dump(TimeLine *time, Sim *s) {
    const OptDump *o = &s->opt.dump;
    if (time_line_cross(time, o->freq_diag))
        dump_diag(time_line_get_current(time), s);
    if (o->parts && time_line_cross(time, o->freq_parts))
        dump_parts(s);
    if (o->mesh && time_line_cross(time, o->freq_mesh))
        dump_mesh(s);
    if (o->field && time_line_cross(time, o->freq_field))
        dump_field(s);
    if (o->strt  && time_line_cross(time, o->freq_strt))
        dump_strt(s);
}
