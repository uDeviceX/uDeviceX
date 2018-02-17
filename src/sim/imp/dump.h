static int download_pp(Sim *s) { /* device to host  data transfer */
    int np = 0;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;

    if (flu->q.n) {
        cD2H(s->pp_dump + np, flu->q.pp, flu->q.n);    np += flu->q.n;
    }
    if (s->solids0 && rig->q.n) {
        cD2H(s->pp_dump + np, rig->q.pp, rig->q.n);    np += rig->q.n;
    }
    if (rbcs && rbc->q.n) {
        cD2H(s->pp_dump + np, rbc->q.pp, rbc->q.n);    np += rbc->q.n;
    }
    return np;
}

static void dump_part(Sim *s) {
    const Flu *flu = &s->flu;
    const Rig *rig = &s->rig;
    BopWork *dumpt = s->dumpt;
    static int id_bop = 0; /* TODO */
    cD2H(flu->q.pp_hst, flu->q.pp, flu->q.n);
    if (s->opt.fluids) {
        cD2H(flu->q.ii_hst, flu->q.ii, flu->q.n);
        bop_ids(s->cart, flu->q.ii_hst, flu->q.n, "id_solvent", id_bop);
    }
    if (s->opt.flucolors) {
        cD2H(flu->q.cc_hst, flu->q.cc, flu->q.n);
        bop_colors(s->cart, flu->q.cc_hst, flu->q.n, "colors_solvent", id_bop);
    }
    if (s->opt.fluss) {
        cD2H(flu->ss_hst, flu->ss, 6 * flu->q.n);
        bop_stresses(s->cart, flu->ss_hst, flu->q.n, "stress_solvent", id_bop);
    }

    if (force_dumps) {
        cD2H(flu->ff_hst, flu->ff, flu->q.n);
        bop_parts_forces(s->cart, s->coords, flu->q.pp_hst, flu->ff_hst, flu->q.n, "solvent", id_bop, /**/ dumpt);
    } else {
        bop_parts(s->cart, s->coords, flu->q.pp_hst, flu->q.n, "solvent", id_bop, /**/ dumpt);
    }

    if(s->solids0) {
        cD2H(rig->q.pp_hst, rig->q.pp, rig->q.n);
        if (force_dumps) {
            cD2H(rig->ff_hst, rig->ff, rig->q.n);
            bop_parts_forces(s->cart, s->coords, rig->q.pp_hst, rig->ff_hst, rig->q.n, "solid", id_bop, /**/ dumpt);
        } else {
            bop_parts(s->cart, s->coords, rig->q.pp_hst, rig->q.n, "solid", id_bop, /**/ dumpt);
        }
    }
    id_bop++;
}

static void dump_rbcs(Sim *s) {
    const Rbc *r = &s->rbc;
    static int id = 0;
    cD2H(s->pp_dump, r->q.pp, r->q.n);
    UC(mesh_write_dump(r->mesh_write, s->cart, s->coords, r->q.nc, s->pp_dump, id++));
}

static void dump_rbc_coms(Sim *s) {
    static int id = 0;
    float3 *rr, *vv;
    Rbc *r = &s->rbc;
    int nc = r->q.nc;
    UC(rbc_com_compute(r->com, nc, r->q.pp, /**/ &rr, &vv));
    UC(dump_com(s->cart, s->coords, id++, nc, r->q.ii, rr));
}

static void dump_grid(const Sim *s) {
    const Flu *flu = &s->flu;
    const Rbc *rbc = &s->rbc;
    const Rig *rig = &s->rig;

    QQ qq; /* pack for io/field_dumps */
    NN nn;
    qq.o = flu->q.pp; qq.s = rig->q.pp; qq.r = rbc->q.pp;
    nn.o = flu->q.n ; nn.s = rig->q.n ;  nn.r = rbc->q.n;
    fields_grid(s->coords, s->cart, qq, nn, /*w*/ s->pp_dump);
}

void dump_diag_after(Time *time, int it, bool solid0, Sim *s) { /* after wall */
    float dt;
    const Rig *rig = &s->rig;
    const Opt *o = &s->opt;
    if (solid0 && (time_cross(time, o->freq_parts))) {
        static int id = 0;
        dt = time_dt(time);
        rig_dump(dt, it, rig->q.ss_dmp, rig->q.ss_dmp_bb, rig->q.ns, s->coords);
        cD2H(s->pp_dump, rig->q.i_pp, rig->q.ns * rig->q.nv);
        UC(mesh_write_dump(rig->mesh_write, s->cart, s->coords, rig->q.ns, s->pp_dump, id++));
    }
}

static void diag(float time, Sim *s) {
    if (time < 0) ERR("time = %g < 0", time);
    int n;
    n = download_pp(s);
    UC(diag(s->cart, time, n, s->pp_dump));
}

static void dump_strt_templ(const Coords *coords, Wall *w, Sim *s) { /* template dumps (wall, solid) */
    Rig *rig = &s->rig;
    if (strt_dumps) {
        if (walls) wall_strt_dump_templ(coords, &w->q);
        if (s->opt.rig) rig_strt_dump_templ(coords, &rig->q);
    }
}

static void dump_strt0(int id, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    flu_strt_dump(s->coords, id, &flu->q);
    if (rbcs)       rbc_strt_dump(s->coords, id, &rbc->q);
    if (s->opt.rig) rig_strt_dump(s->coords, id, &rig->q);
}

static void dump_strt(Sim *s) {
    static int id = 0;
    dump_strt0(id++, s);
}

static void dump_diag(Time *time, int it, Sim *s) {
    const Opt *o = &s->opt;
    if (time_cross(time, o->freq_parts)) {
        if (o->dump_parts) dump_part(s);
        if (rbcs)          dump_rbcs(s);
        UC(diag(time_current(time), s));
    }
    if (o->dump_field && time_cross(time, o->freq_field))
        dump_grid(s);
    if (strt_dumps  && it % strt_freq == 0)
        dump_strt(s);
    if (rbc_com_dumps && it % rbc_com_freq == 0)
        dump_rbc_coms(s);
}
