/* see bund.cu for more sim:: functions */
void create_walls() {
    int nold = o::q.n;
    wall::gen_quants(w::qsdf, /**/ &o::q.n, o::q.pp, &w::q);
    o::q.cells->build(o::q.pp, o::q.n);
    MSG("solvent particles survived: %d/%d", o::q.n, nold);
}

void create_solids() {
    cD2H(o::q.pp_hst, o::q.pp, o::q.n);
    rig::gen_quants(/*io*/ o::q.pp_hst, &o::q.n, /**/ &s::q);
    MC(l::m::Barrier(l::m::cart));
    cH2D(o::q.pp, o::q.pp_hst, o::q.n);
    MC(l::m::Barrier(l::m::cart));
    MSG("created %d solids.", s::q.ns);
}

void freeze(rbc::Quants *qrbc, sdf::Quants qsdf) {
    MC(l::m::Barrier(l::m::cart));
    if (solids)           create_solids();
    if (walls && rbcs  )  remove_rbcs(qrbc, qsdf);
    if (walls && solids)  remove_solids();
    if (solids)           rig::set_ids(s::q);
}
