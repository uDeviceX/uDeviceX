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

void forces_rbc (Rbc *r) {
    rbcforce_apply(r->q, r->tt, /**/ r->ff);
    if (RBC_STRETCH) rbc::stretch::rbc_stretch_apply(r->q.nc, r->stretch, /**/ r->ff);
}

void clear_forces(Force* ff, int n) {
    if (n) DzeroA(ff, n);
}

void forces_wall(Wall *w, Sim *s) {
    using namespace wall;
    Cloud co, cs, cr;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    ini_cloud(flu->q.pp, &co);
    ini_cloud(rig->q.pp, &cs);
    ini_cloud(rbc->q.pp, &cr);
    if (multi_solvent) ini_cloud_color(flu->q.cc, &co);
    
    if (flu->q.n)               color::force(w->vview, s->coords, w->sdf, w->q, w->t, co, flu->q.n, /**/ flu->ff);
    if (s->solids0 && rig->q.n) grey::force(w->vview, s->coords, w->sdf, w->q, w->t, cs, rig->q.n, /**/ rig->ff);
    if (rbcs && rbc->q.n)       grey::force(w->vview, s->coords, w->sdf, w->q, w->t, cr, rbc->q.n, /**/ rbc->ff);
}
