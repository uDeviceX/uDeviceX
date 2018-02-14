void body_force(long it, const BForce *bforce, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;

    if (pushflu)
        UC(bforce_apply(it, s->coords, flu_mass, bforce, flu->q.n, flu->q.pp, /**/ flu->ff));
    if (pushsolid && s->solids0)
        UC(bforce_apply(it, s->coords, solid_mass, bforce, rig->q.n, rig->q.pp, /**/ rig->ff));
    if (pushrbc && rbcs)
        UC(bforce_apply(it, s->coords, rbc_mass, bforce, rbc->q.n, rbc->q.pp, /**/ rbc->ff));
}

void forces_rbc (float dt, Rbc *r) {
    rbc_force_apply(r->force, r->params, dt, &r->q, /**/ r->ff);
    if (RBC_STRETCH) rbc_stretch_apply(r->q.nc, r->stretch, /**/ r->ff);
}

void clear_forces(Force* ff, int n) {
    if (n) DzeroA(ff, n);
}

void forces_wall(Wall *w, Sim *s) {
    PaArray po, ps, pr;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    PairParams *par = flu->params;
    parray_push_pp(flu->q.pp, &po);
    parray_push_pp(rig->q.pp, &ps);
    parray_push_pp(rbc->q.pp, &pr);
    if (multi_solvent)
        parray_push_cc(flu->q.cc, &po);
    
    if (flu->q.n)               wall_force(par, w->vview, s->coords, w->sdf, &w->q, w->t, flu->q.n, &po, /**/ flu->ff);
    if (s->solids0 && rig->q.n) wall_force(par, w->vview, s->coords, w->sdf, &w->q, w->t, rig->q.n, &ps, /**/ rig->ff);
    if (rbcs && rbc->q.n)       wall_force(par, w->vview, s->coords, w->sdf, &w->q, w->t, rbc->q.n, &pr, /**/ rbc->ff);
}
