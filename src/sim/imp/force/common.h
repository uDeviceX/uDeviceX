void body_force(const BForce *bforce, Sim *s) {
    Flu *flu = &s->flu;
    const Opt *opt = &s->opt;

    if (opt->push_flu)
        UC(bforce_apply(s->coords, flu->mass, bforce, flu->q.n, flu->q.pp, /**/ flu->ff));
    UC(objects_body_forces(bforce, s->obj));
}

void forces_rbc (float dt, const Opt *o, Rbc *r) {
    rbc_force_apply(r->force, r->params, dt, &r->q, /**/ r->ff);
    if (o->rbc.stretch) rbc_stretch_apply(r->q.nc, r->stretch, /**/ r->ff);
}

static void clear_forces(int n, Force* ff) {
    if (n) DzeroA(ff, n);
}

static void clear_stresses(int n, float* ss) {
    if (n) DzeroA(ss, 6*n);
}

void forces_wall(bool fluss, Sim *s) {
    PaArray po;
    FoArray fo;
    PFarrays *pf;
    Flu *flu = &s->flu;
    Wall *w =  s->wall;
    const PairParams *par = flu->params;
    const Opt *opt = &s->opt;

    if (!w) return;
    
    parray_push_pp(flu->q.pp, &po);
    if (opt->flucolors)
        parray_push_cc(flu->q.cc, &po);

    farray_push_ff(flu->ff, &fo);
    if (fluss)
        farray_push_ss(flu->ss, &fo);

    UC(pfarrays_ini(&pf));
    UC(pfarrays_push(pf, flu->q.n, po, fo));
    UC(objects_get_particles_all(s->obj, pf));
    UC(wall_interact(s->coords, par, w, pf));
    UC(pfarrays_fin(pf));
    
    // if (active_rig(s) && rig->q.n) wall_repulse(rig->q.n, rig->q.pp, w->sdf, /**/ rig->ff);
    // if (active_rbc(s) && rbc->q.n) wall_repulse(rbc->q.n, rbc->q.pp, w->sdf, /**/ rbc->ff);
}
