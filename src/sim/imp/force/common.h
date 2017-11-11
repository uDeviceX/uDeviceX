void body_force(scheme::Fparams fpar) {
    scheme::force(1, fpar,  o::q.n, o::q.pp, /**/ o::ff);
    if (pushsolid && solids0)
        scheme::force(solid_mass, fpar, s::q.n, s::q.pp, /**/ s::ff);
    if (pushrbc && rbcs)
        scheme::force(rbc_mass, fpar, r::q.n, r::q.pp, /**/ r::ff);
}

void forces_rbc() {
    if (rbcs)
        rbc::force::forces(r::q, r::tt, /**/ r::ff);
}

void clear_forces(Force* ff, int n) {
    if (n) DzeroA(ff, n);
}

void forces_wall() {
    using namespace wall;
    Cloud co, cs, cr;
    ini_cloud(o::q.pp, &co);
    ini_cloud(s::q.pp, &cs);
    ini_cloud(r::q.pp, &cr);
    if (multi_solvent) ini_cloud_color(o::q.cc, &co);
    
    if (o::q.n)           color::force(w::qsdf, w::q, w::t, co, o::q.n, /**/ o::ff);
    if (solids0 && s::q.n) grey::force(w::qsdf, w::q, w::t, cs, s::q.n, /**/ s::ff);
    if (rbcs && r::q.n)    grey::force(w::qsdf, w::q, w::t, cr, r::q.n, /**/ r::ff);
}
