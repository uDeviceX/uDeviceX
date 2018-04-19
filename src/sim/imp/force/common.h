void body_force(const BForce *bforce, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    const Opt *opt = &s->opt;

    if (opt->push_flu)
        UC(bforce_apply(s->coords, flu->mass, bforce, flu->q.n, flu->q.pp, /**/ flu->ff));
    if (opt->push_rig && active_rig(s))
        UC(bforce_apply(s->coords, rig->mass, bforce, rig->q.n, rig->q.pp, /**/ rig->ff));
    if (opt->rbc.push && active_rbc(s))
        UC(bforce_apply(s->coords, rbc->mass, bforce, rbc->q.n, rbc->q.pp, /**/ rbc->ff));
}

void forces_rbc (float dt, const Opt *o, Rbc *r) {
    rbc_force_apply(r->force, r->params, dt, &r->q, /**/ r->ff);
    if (o->rbc.stretch) rbc_stretch_apply(r->q.nc, r->stretch, /**/ r->ff);
}

void clear_forces(int n, Force* ff) {
    if (n) DzeroA(ff, n);
}

void clear_stresses(int n, float* ss) {
    if (n) DzeroA(ss, 6*n);
}

void forces_wall(bool fluss, Sim *s) {
    PaArray po, ps, pr;
    FoArray fo, fs, fr;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *w = &s->wall;
    PairParams *par = flu->params;
    const Opt *opt = &s->opt;

    if (!w->q.n) return;
    
    parray_push_pp(flu->q.pp, &po);
    parray_push_pp(rig->q.pp, &ps);
    parray_push_pp(rbc->q.pp, &pr);
    if (opt->flucolors)
        parray_push_cc(flu->q.cc, &po);

    farray_push_ff(flu->ff, &fo);
    farray_push_ff(rig->ff, &fs);
    farray_push_ff(rbc->ff, &fr);
    if (fluss)
        farray_push_ss(flu->ss, &fo);
    
    if (flu->q.n)                  wall_force(par, w->velstep, s->coords, w->sdf, &w->q, w->t, flu->q.n, &po, /**/ &fo);
    if (active_rig(s) && rig->q.n) wall_force(par, w->velstep, s->coords, w->sdf, &w->q, w->t, rig->q.n, &ps, /**/ &fs);
    if (active_rbc(s) && rbc->q.n) wall_force(par, w->velstep, s->coords, w->sdf, &w->q, w->t, rbc->q.n, &pr, /**/ &fr);

    // if (active_rig(s) && rig->q.n) wall_repulse(rig->q.n, rig->q.pp, w->sdf, /**/ rig->ff);
    // if (active_rbc(s) && rbc->q.n) wall_repulse(rbc->q.n, rbc->q.pp, w->sdf, /**/ rbc->ff);
}
