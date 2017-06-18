namespace s {
void load_solid_mesh(const char *fname) {
    ply::read(fname, &m_hst);

    m_dev.nv = m_hst.nv;
    m_dev.nt = m_hst.nt;

    CC(cudaMalloc(&(m_dev.tt), 3 * m_dev.nt * sizeof(int)));
    CC(cudaMalloc(&(m_dev.vv), 3 * m_dev.nv * sizeof(float)));

    cH2D(m_dev.tt, m_hst.tt, 3 * m_dev.nt);
    cH2D(m_dev.vv, m_hst.vv, 3 * m_dev.nv);
}

void allocate() {
    mpDeviceMalloc(&pp);
    mpDeviceMalloc(&ff);

    CC(cudaMalloc(&ss_dev,    MAX_SOLIDS * sizeof(Solid)));
    CC(cudaMalloc(&ss_bb_dev, MAX_SOLIDS * sizeof(Solid)));

    ss_hst      = new Solid[MAX_SOLIDS];
    ss_bb_hst   = new Solid[MAX_SOLIDS];
    ss_dmphst   = new Solid[MAX_SOLIDS];
    ss_dmpbbhst = new Solid[MAX_SOLIDS];

    i_pp_hst    = new Particle[MAX_PART_NUM];
    i_pp_bb_hst = new Particle[MAX_PART_NUM];
    CC(cudaMalloc(   &i_pp_dev, MAX_PART_NUM * sizeof(Particle)));
    CC(cudaMalloc(&i_pp_bb_dev, MAX_PART_NUM * sizeof(Particle)));

    minbb_hst = new float3[MAX_SOLIDS];
    maxbb_hst = new float3[MAX_SOLIDS];
    CC(cudaMalloc(&minbb_dev, MAX_SOLIDS * sizeof(float3)));
    CC(cudaMalloc(&maxbb_dev, MAX_SOLIDS * sizeof(float3)));
}

void allocate_tcells() {
    tcs_hst = new int[XS * YS * ZS];
    tcc_hst = new int[XS * YS * ZS];
    tci_hst = new int[27 * MAX_SOLIDS * m_hst.nt]; // assume 1 triangle don't overlap more than 27 cells

    
    CC(cudaMalloc(&tcs_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&tcc_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&tci_dev, 27 * MAX_SOLIDS * m_dev.nt * sizeof(int)));
}

void deallocate() {
    delete[] m_hst.tt;      CC(cudaFree(m_dev.tt));
	delete[] m_hst.vv;      CC(cudaFree(m_dev.vv));

	delete[] tcs_hst;       CC(cudaFree(tcs_dev));
	delete[] tcc_hst;       CC(cudaFree(tcc_dev));
	delete[] tci_hst;       CC(cudaFree(tci_dev));

	delete[] i_pp_hst;      CC(cudaFree(i_pp_dev));
	delete[] i_pp_bb_hst;   CC(cudaFree(i_pp_bb_dev));

	delete[] minbb_hst;     CC(cudaFree(minbb_dev));
    delete[] maxbb_hst;     CC(cudaFree(maxbb_dev));
	delete[] ss_hst;        CC(cudaFree(ss_dev));
	delete[] ss_bb_hst;     CC(cudaFree(ss_bb_dev));
	delete[] ss_dmphst;     delete[] ss_dmpbbhst;
}

void create(Particle *opp, int *on) {
    load_solid_mesh("mesh_solid.ply");

    // generate models
    MSG("start solid ini");
    ic::ini("ic_solid.txt", m_hst, /**/ &ns, &nps, rr0_hst, ss_hst, on, opp, pp_hst);
    MSG("done solid ini");

    allocate_tcells();
    
    // generate the solid particles

    solid::generate_hst(ss_hst, ns, rr0_hst, nps, /**/ pp_hst);
    solid::reinit_ft_hst(ns, /**/ ss_hst);
    npp = ns * nps;

    solid::mesh2pp_hst(ss_hst, ns, m_hst, /**/ i_pp_hst);
    cH2D(i_pp_dev, i_pp_hst, ns * m_hst.nv);

    cH2D(ss_dev, ss_hst, ns);
    cH2D(rr0, rr0_hst, 3 * nps);

    cH2D(pp, pp_hst, npp);

    MC(l::m::Barrier(m::cart));
}

void ini() {
    npp = ns = nps = 0;
    allocate();
}

void fin() {
    deallocate();
}
}
