static void dev2hst(Sim *s) { /* device to host  data transfer */
    int start = 0;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    
    cD2H(s->pp_dump + start, flu->q.pp, flu->q.n); start += flu->q.n;
    if (s->solids0) {
        cD2H(s->pp_dump + start, rig->q.pp, rig->q.n); start += rig->q.n;
    }
    if (rbcs) {
        cD2H(s->pp_dump + start, rbc->q.pp, rbc->q.n); start += rbc->q.n;
    }
}

static void dump_part(int step, Sim *s) {
    const Flu *flu = &s->flu;
    const Rig *rig = &s->rig;
    BopWork *dumpt = s->dumpt;
    int id_bop = step / s->opt.freq_parts;
    
    cD2H(flu->q.pp_hst, flu->q.pp, flu->q.n);
    if (global_ids) {
        cD2H(flu->q.ii_hst, flu->q.ii, flu->q.n);
        bop_ids(s->cart, flu->q.ii_hst, flu->q.n, "id_solvent", id_bop);
    }
    if (multi_solvent) {
        cD2H(flu->q.cc_hst, flu->q.cc, flu->q.n);
        bop_colors(s->cart, flu->q.cc_hst, flu->q.n, "colors_solvent", id_bop);
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

void dump_diag_after(float dt, int it, bool solid0, Sim *s) { /* after wall */
    const Rig *rig = &s->rig;
    const Opt *o = &s->opt;
    if (solid0 && it % o->freq_parts == 0) {
        static int id = 0;
        rig_dump(dt, it, rig->q.ss_dmp, rig->q.ss_dmp_bb, rig->q.ns, s->coords);
        cD2H(s->pp_dump, rig->q.i_pp, rig->q.ns * rig->q.nv);
        UC(mesh_write_dump(rig->mesh_write, s->cart, s->coords, rig->q.ns, s->pp_dump, id++));
    }
}

static void diag(float dt, int it, Sim *s) {
    const Flu *flu = &s->flu;
    const Rbc *rbc = &s->rbc;
    const Rig *rig = &s->rig;
    int n = flu->q.n;
    if (rbcs)       n += rbc->q.n;
    if (s->solids0) n += rig->q.n;
    dev2hst(s);
    diagnostics(dt, s->cart, n, s->pp_dump, it);
}

void dump_strt_templ(const Coords *coords, Wall *w, Sim *s) { /* template dumps (wall, solid) */
    Rig *rig = &s->rig;
    if (strt_dumps) {
        if (walls) wall_strt_dump_templ(coords, &w->q);
        if (solids) rig_strt_dump_templ(coords, &rig->q);
    }
}

void dump_strt(int id, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    flu_strt_dump(s->coords, id, &flu->q);
    if (rbcs)       rbc_strt_dump(s->coords, id, &rbc->q);
    if (solids)     rig_strt_dump(s->coords, id, &rig->q);
}

void dump_diag0(float dt, int it, Sim *s) { /* generic dump */
    const Opt *o = &s->opt;

    if (it % o->freq_parts == 0) {
        if (o->dump_parts) dump_part(it, s);
        if (rbcs)          dump_rbcs(s);
        diag(dt, it, s);
    }
    if (o->dump_field && it % o->freq_field == 0) dump_grid(s);
    if (strt_dumps  && it % strt_freq == 0)  dump_strt(it / strt_freq, s);
    if (rbc_com_dumps && it % rbc_com_freq == 0) dump_rbc_coms(s);
}
