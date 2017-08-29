/* see bund.cu for more sim:: functions */
void create_walls() {
    int nold = o::q.n;
    wall::gen_quants(w::qsdf, /**/ &o::q.n, o::q.pp, &w::q);
    o::q.cells->build(o::q.pp, o::q.n);
    MSG("solvent particles survived: %d/%d", o::q.n, nold);
}

void create_solids(flu::Quants* qflu, rig::Quants* qrig) {
    cD2H(qflu->pp_hst, qflu->pp, qflu->n);
    rig::gen_quants(/*io*/ qflu->pp_hst, &qflu->n, /**/ qrig);
    MC(l::m::Barrier(l::m::cart));
    cH2D(qflu->pp, qflu->pp_hst, qflu->n);
    MC(l::m::Barrier(l::m::cart));
    MSG("created %d solids.", qrig->ns);
}

void freeze(flu::Quants *qflu, rig::Quants *qrig, rbc::Quants *qrbc, sdf::Quants qsdf) {
    MC(l::m::Barrier(l::m::cart));
    if (solids)           create_solids(qflu, qrig);
    if (walls && rbcs  )  remove_rbcs(qrbc, qsdf);
    if (walls && solids)  remove_solids(qrig, qsdf);
    if (solids)           rig::set_ids(*qrig);
}
