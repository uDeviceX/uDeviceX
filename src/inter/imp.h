namespace inter {
void create_walls(sdf::Quants qsdf, flu::Quants* qflu, wall::Quants *qwall) {
    int nold = qflu->n;
    wall::gen_quants(qsdf, /**/ &qflu->n, qflu->pp, qwall);
    qflu->cells->build(qflu->pp, qflu->n);
    MSG("solvent particles survived: %d/%d", qflu->n, nold);
}

void freeze(sdf::Quants qsdf, flu::Quants *qflu, rig::Quants *qrig, rbc::Quants *qrbc) {
    MC(l::m::Barrier(l::m::cart));
    if (solids)           create_solids(qflu, qrig);
    if (walls && rbcs  )  remove_rbcs(qrbc, qsdf);
    if (walls && solids)  remove_solids(qrig, qsdf);
    if (solids)           rig::set_ids(*qrig);
}
} /* namespace */
