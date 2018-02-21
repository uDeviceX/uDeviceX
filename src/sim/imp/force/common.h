void body_force(long it, const BForce *bforce, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Opt *opt = &s->opt;

    if (opt->push_flu)
        UC(bforce_apply(it, s->coords, flu->mass, bforce, flu->q.n, flu->q.pp, /**/ flu->ff));
    if (opt->push_rig && s->solids0)
        UC(bforce_apply(it, s->coords, rig->mass, bforce, rig->q.n, rig->q.pp, /**/ rig->ff));
    if (opt->push_rbc && s->opt.rbc)
        UC(bforce_apply(it, s->coords, rbc->mass, bforce, rbc->q.n, rbc->q.pp, /**/ rbc->ff));
}

void forces_rbc (float dt, Rbc *r) {
    rbc_force_apply(r->force, r->params, dt, &r->q, /**/ r->ff);
    if (RBC_STRETCH) rbc_stretch_apply(r->q.nc, r->stretch, /**/ r->ff);
}

void clear_forces(Force* ff, int n) {
    if (n) DzeroA(ff, n);
}

void clear_stresses(float* ss, int n) {
    if (n) DzeroA(ss, 6*n);
}

void forces_wall(bool fluss, Wall *w, Sim *s) {
    PaArray po, ps, pr;
    FoArray fo, fs, fr;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    PairParams *par = flu->params;
    parray_push_pp(flu->q.pp, &po);
    parray_push_pp(rig->q.pp, &ps);
    parray_push_pp(rbc->q.pp, &pr);
    if (s->opt.flucolors)
        parray_push_cc(flu->q.cc, &po);

    farray_push_ff(flu->ff, &fo);
    farray_push_ff(rig->ff, &fs);
    farray_push_ff(rbc->ff, &fr);
    if (fluss)
        farray_push_ss(flu->ss, &fo);
    
    if (flu->q.n)               wall_force(par, w->velstep, s->coords, w->sdf, &w->q, w->t, flu->q.n, &po, /**/ &fo);
    if (s->solids0 && rig->q.n) wall_force(par, w->velstep, s->coords, w->sdf, &w->q, w->t, rig->q.n, &ps, /**/ &fs);
    if (s->opt.rbc && rbc->q.n) wall_force(par, w->velstep, s->coords, w->sdf, &w->q, w->t, rbc->q.n, &pr, /**/ &fr);
}
