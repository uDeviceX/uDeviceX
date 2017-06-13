void load_solid_mesh(const char *fname) {
    ply::read(fname, &m_hst);

    m_dev.nv = m_hst.nv;
    m_dev.nt = m_hst.nt;

    CC(cudaMalloc(&(m_dev.tt), 3 * m_dev.nt * sizeof(int)));
    CC(cudaMalloc(&(m_dev.vv), 3 * m_dev.nv * sizeof(float)));

    CC(cudaMemcpy(m_dev.tt, m_hst.tt, 3 * m_dev.nt * sizeof(int), H2D));
    CC(cudaMemcpy(m_dev.vv, m_hst.vv, 3 * m_dev.nv * sizeof(float), H2D));
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

    bboxes_hst = new float[6*MAX_SOLIDS];
    CC(cudaMalloc(&bboxes_dev, 6*MAX_SOLIDS * sizeof(float)));
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

	delete[] bboxes_hst;    CC(cudaFree(bboxes_dev));
	delete[] ss_hst;        CC(cudaFree(ss_dev));
	delete[] ss_bb_hst;     CC(cudaFree(ss_bb_dev));
	delete[] ss_dmphst;     delete[] ss_dmpbbhst;
}

void create(Particle *opp, int *on) {
    load_solid_mesh("mesh_solid.ply");

    // generate models
    MSG0("start solid init");
    ic::init("ic_solid.txt", m_hst, /**/ &ns, &nps, rr0_hst, ss_hst, on, opp, pp_hst);
    MSG0("done solid init");

    allocate_tcells();
    
    // generate the solid particles

    solid::generate_hst(ss_hst, ns, rr0_hst, nps, /**/ pp_hst);
    solid::reinit_ft_hst(ns, /**/ ss_hst);
    npp = ns * nps;

    solid::mesh2pp_hst(ss_hst, ns, m_hst, /**/ i_pp_hst);
    CC(cudaMemcpy(i_pp_dev, i_pp_hst, ns * m_hst.nv * sizeof(Particle), H2D));

    CC(cudaMemcpy(ss_dev, ss_hst, ns * sizeof(Solid), H2D));
    CC(cudaMemcpy(rr0, rr0_hst, 3 * nps * sizeof(float), H2D));

    CC(cudaMemcpy(pp, pp_hst, sizeof(Particle) * npp, H2D));

    MC(MPI_Barrier(m::cart));
}

void ini() {
    npp = ns = nps = 0;
    allocate();
}

void fin() {
    deallocate();
}
