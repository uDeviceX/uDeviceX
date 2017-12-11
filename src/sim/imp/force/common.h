void body_force(FParam fpar) {
    if (pushflu)
        body_force(coords, flu_mass, fpar,  flu.q.n, flu.q.pp, /**/ flu.ff);
    if (pushsolid && solids0)
        body_force(coords, solid_mass, fpar, rig.q.n, rig.q.pp, /**/ rig.ff);
    if (pushrbc && rbcs)
        body_force(coords, rbc_mass, fpar, rbc.q.n, rbc.q.pp, /**/ rbc.ff);
}

void forces_rbc (Rbc *r) {
    rbc::force::apply(r->q, r->tt, /**/ r->ff);
    if (RBC_STRETCH) rbc::stretch::apply(r->q.nc, r->stretch, /**/ r->ff);
}

void clear_forces(Force* ff, int n) {
    if (n) DzeroA(ff, n);
}

void forces_wall(Wall *w) {
    using namespace wall;
    Cloud co, cs, cr;
    ini_cloud(flu.q.pp, &co);
    ini_cloud(rig.q.pp, &cs);
    ini_cloud(rbc.q.pp, &cr);
    if (multi_solvent) ini_cloud_color(flu.q.cc, &co);
    
    if (flu.q.n)           color::force(w->vel, coords, w->sdf, w->q, w->t, co, flu.q.n, /**/ flu.ff);
    if (solids0 && rig.q.n) grey::force(w->vel, coords, w->sdf, w->q, w->t, cs, rig.q.n, /**/ rig.ff);
    if (rbcs && rbc.q.n)    grey::force(w->vel, coords, w->sdf, w->q, w->t, cr, rbc.q.n, /**/ rbc.ff);
}
