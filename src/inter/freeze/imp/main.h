template <typename T>
static void remove(T *data, int nv, int *e, int nc) {
    int c; /* c: cell */
    for (c = 0; c < nc; c++) cA2A(data + nv*c, data + nv*e[c], nv);
}

static void remove_rbcs(RbcQuants *q, Sdf *qsdf) {
    int stay[MAX_CELL_NUM];
    int nc0;
    q->nc = sdf_who_stays(qsdf, q->n, q->pp, nc0 = q->nc, q->nv, /**/ stay);
    q->n = q->nc * q->nv;
    remove(q->pp, q->nv, stay, q->nc);
    msg_print("%d/%d RBCs survived", q->nc, nc0);
}

static void create_solids(const Coords *coords, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo *pi,
                          MPI_Comm cart, FluQuants* qflu, RigQuants* qrig) {
    cD2H(qflu->pp_hst, qflu->pp, qflu->n);
    rig_gen_quants(coords, empty_pp, numdensity, rig_mass, pi, cart, /*io*/ qflu->pp_hst, &qflu->n, /**/ qrig);
    MC(m::Barrier(cart));
    cH2D(qflu->pp, qflu->pp_hst, qflu->n);
    MC(m::Barrier(cart));
    msg_print("created %d solids.", qrig->ns);
}

static void remove_solids(RigQuants *q, Sdf *sdf) {
    int stay[MAX_SOLIDS];
    int ns0;
    int nip = q->ns * q->nv;
    q->ns = sdf_who_stays(sdf, nip, q->i_pp, ns0 = q->ns, q->nv, /**/ stay);
    q->n  = q->ns * q->nps;
    remove(q->pp,       q->nps,      stay, q->ns);
    remove(q->pp_hst,   q->nps,      stay, q->ns);

    remove(q->ss,       1,           stay, q->ns);
    remove(q->ss_hst,   1,           stay, q->ns);

    remove(q->i_pp,     q->nv, stay, q->ns);
    remove(q->i_pp_hst, q->nv, stay, q->ns);
    msg_print("sim.impl: %d/%d Solids survived", q->ns, ns0);
}

void inter_freeze_walls(MPI_Comm cart, int maxn, Sdf *sdf, FluQuants* qflu, WallQuants *qwall) {
    int nold = qflu->n;
    UC(wall_gen_quants(cart, maxn, sdf, /**/ &qflu->n, qflu->pp, qwall));
    flu_build_cells(qflu);
    msg_print("solvent particles survived: %d/%d", qflu->n, nold);
}

void inter_freeze(const Coords *coords, MPI_Comm cart, InterWalInfos w, InterFluInfos f, InterRbcInfos r, InterRigInfos s) {
    MC(m::Barrier(cart));
    if (s.active)             create_solids(coords, s.empty_pp, s.numdensity, s.mass, s.pi, cart, f.q, s.q);
    if (w.active && r.active) remove_rbcs(r.q, w.sdf);
    if (w.active && s.active) remove_solids(s.q, w.sdf);
    if (s.active)             rig_set_ids(cart, s.q);
}
