static char buf[BUFSIZ];
#define F(s) fmsg(s, __FILE__, __LINE__)
static const char *fmsg(const char *msg, const char *f, int n) {
    sprintf(buf, "%s:%d: %s", f, n, msg);
    return buf;
}

void body_force(float driving_force0) {
    if (pushflow)
        KL(dev::body_force, (k_cnf(o::q.n)), (1, o::q.pp, o::ff, o::q.n, driving_force0));

    if (pushsolid && solids0)
        KL(dev::body_force, (k_cnf(s::q.n)), (solid_mass, s::q.pp, s::ff, s::q.n, driving_force0));

    if (pushrbc && rbcs)
        KL(dev::body_force, (k_cnf(r::q.n)), (rbc_mass, r::q.pp, r::ff, r::q.n, driving_force0));
}

void forces_rbc() {
    if (rbcs)
        rbc::forces(r::q, r::tt, /**/ r::ff);
}

void clear_forces(Force* ff, int n) {
    if (n) DzeroA(ff, n);
}

void forces_wall() {
    dbg::check_ff(o::ff, o::q.n, F("B"));
    if (o::q.n)            wall::interactions(w::qsdf, w::q, w::t, SOLVENT_TYPE, o::q.pp, o::q.n,   /**/ o::ff);
    if (solids0 && s::q.n) wall::interactions(w::qsdf, w::q, w::t, SOLID_TYPE, s::q.pp, s::q.n, /**/ s::ff);
    if (rbcs && r::q.n)    wall::interactions(w::qsdf, w::q, w::t, SOLID_TYPE, r::q.pp, r::q.n, /**/ r::ff);
    dbg::check_ff(o::ff, o::q.n, F("A"));
}

void forces_cnt(std::vector<ParticlesWrap> *w_r) {
    dbg::check_ff(o::ff, o::q.n, F("B"));
    cnt::build(*w_r); /* build cells */
    cnt::bulk(*w_r);
    dbg::check_ff(o::ff, o::q.n, F("A"));
}

void forces_fsi(SolventWrap *w_s, std::vector<ParticlesWrap> *w_r) {
    dbg::check_ff(o::ff, o::q.n, F("B"));
    fsi::bind(*w_s);
    fsi::bulk(*w_r);
    dbg::check_ff(o::ff, o::q.n, F("A"));
}
