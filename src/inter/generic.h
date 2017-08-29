template <typename T>
static void remove(T *data, int nv, int *e, int nc) {
    int c; /* c: cell */
    for (c = 0; c < nc; c++) cA2A(data + nv*c, data + nv*e[c], nv);
}

void remove_rbcs(rbc::Quants *q, sdf::Quants qsdf) {
    int stay[MAX_CELL_NUM];
    int nc0;
    q->nc = sdf::who_stays(qsdf, q->pp, q->n, nc0 = q->nc, q->nv, /**/ stay);
    q->n = q->nc * q->nv;
    remove(q->pp, q->nv, stay, q->nc);
    MSG("%d/%d RBCs survived", q->nc, nc0);
}

void remove_solids() {
    int stay[MAX_SOLIDS];
    int ns0;
    int nip = s::q.ns * s::q.m_dev.nv;
    s::q.ns = sdf::who_stays(w::qsdf, s::q.i_pp, nip, ns0 = s::q.ns, s::q.m_dev.nv, /**/ stay);
    s::q.n  = s::q.ns * s::q.nps;
    remove(s::q.pp,       s::q.nps,      stay, s::q.ns);
    remove(s::q.pp_hst,   s::q.nps,      stay, s::q.ns);

    remove(s::q.ss,       1,           stay, s::q.ns);
    remove(s::q.ss_hst,   1,           stay, s::q.ns);

    remove(s::q.i_pp,     s::q.m_dev.nv, stay, s::q.ns);
    remove(s::q.i_pp_hst, s::q.m_hst.nv, stay, s::q.ns);
    MSG("sim.impl: %d/%d Solids survived", s::q.ns, ns0);
}

