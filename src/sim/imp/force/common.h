void body_force(scheme::force::Param fpar) {
    scheme::force::main(1, fpar,  flu.q.n, flu.q.pp, /**/ flu.ff);
    if (pushsolid && solids0)
        scheme::force::main(solid_mass, fpar, s::q.n, s::q.pp, /**/ s::ff);
    if (pushrbc && rbcs)
        scheme::force::main(rbc_mass, fpar, r::q.n, r::q.pp, /**/ r::ff);
}

void forces_rbc () {
    rbc::force::apply(r::q, r::tt, /**/ r::ff);
    if (RBC_STRETCH) rbc::stretch::apply(r::q.nc, r::stretch, /**/ r::ff);
}

void clear_forces(Force* ff, int n) {
    if (n) DzeroA(ff, n);
}

void forces_wall() {
    using namespace wall;
    Cloud co, cs, cr;
    ini_cloud(flu.q.pp, &co);
    ini_cloud(s::q.pp, &cs);
    ini_cloud(r::q.pp, &cr);
    if (multi_solvent) ini_cloud_color(flu.q.cc, &co);
    
    if (flu.q.n)           color::force(w::qsdf, w::q, w::t, co, flu.q.n, /**/ flu.ff);
    if (solids0 && s::q.n) grey::force(w::qsdf, w::q, w::t, cs, s::q.n, /**/ s::ff);
    if (rbcs && r::q.n)    grey::force(w::qsdf, w::q, w::t, cr, r::q.n, /**/ r::ff);
}
