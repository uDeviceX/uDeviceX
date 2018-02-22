static void gen0(const Coords *c, MPI_Comm comm, RigGenInfo rgi, int nsolid, int rcount, int idmax, int root, float *coms, /**/
                 RigInfo ri) {
    Solid model;
    // share model to everyone
    int npsolid = 0, rank;
    int3 L = subdomain(c);
    
    MC(m::Comm_rank(comm, &rank));
    if (rank == root) {
        npsolid = rcount;
        if (!npsolid) ERR("No particles remaining in root node.\n");
        for (int d = 0; d < 3; ++d)
            model.com[d] = coms[idmax*3 + d];
        ini_props(rgi.pi, npsolid, ri.pp, rgi.mass, model.com, rgi.nt, rgi.tt, rgi.vv, /**/ ri.rr0, &model);
        if (rgi.empty_pp)
            empty_solid(rgi.nt, rgi.tt, rgi.vv, /* io */ ri.rr0, &npsolid);
    }

    MC(m::Bcast(&npsolid,         1,   MPI_INT, root, comm) );
    MC(m::Bcast(ri.rr0, 3 * npsolid, MPI_FLOAT, root, comm) );
    MC(m::Bcast(&model,    1, datatype::solid,   root, comm) );

    // filter coms to keep only the ones in my domain
    int id = 0;
    for (int j = 0; j < nsolid; ++j) {
        const float *com = coms + 3*j;

        if (-L.x/2 <= com[X] && com[X] < L.x/2 &&
            -L.y/2 <= com[Y] && com[Y] < L.y/2 &&
            -L.z/2 <= com[Z] && com[Z] < L.z/2 ) {
            ri.ss[id] = model;

            for (int d = 0; d < 3; ++d)
                ri.ss[id].com[d] = com[d];

            ++id;
        }
    }

    *ri.ns = nsolid = id;
    *ri.nps = npsolid;

    set_rig_ids(comm, nsolid, /**/ ri.ss);
}

static void gen1(const Coords *coords, MPI_Comm comm, RigGenInfo rgi, int nsolid, float *coms, /**/
                 FluInfo fluinfo, RigInfo riginfo,
                 /*w*/ int *tags, int *rcounts) {
    int root, idmax;
    elect(comm, rcounts, nsolid, /**/ &root, &idmax);
    MC(m::Bcast(&idmax, 1, MPI_INT, root, comm));

    int rcount = 0;
    kill(idmax, tags, /**/ fluinfo.n, fluinfo.pp, &rcount, riginfo.pp);
    DBG("after kill: %d", rcount);

    share(coords, comm, root, /**/ riginfo.pp, &rcount);
    DBG("after share: %d", rcount);

    gen0(coords, comm, rgi, nsolid, rcount, idmax, root, coms, /**/ riginfo);
}

static void gen2(const Coords *coords, MPI_Comm comm, RigGenInfo rgi, int nsolid, float *coms, /**/
                 FluInfo fluinfo, RigInfo riginfo, /*w*/ int *tags, int *rcounts) {
    int3 L = subdomain(coords);
    count_pp_inside(rgi.pi, L, fluinfo.pp, *fluinfo.n, coms, nsolid, rgi.tt, rgi.vv, rgi.nt, /**/ tags, rcounts);
    gen1(coords, comm, rgi, nsolid, coms, /**/ fluinfo, riginfo, /*w*/ tags, rcounts);
}

static void gen(const Coords *coords, MPI_Comm comm, const char *fname, RigGenInfo rgi, /**/
                 FluInfo fluinfo, RigInfo riginfo) {
    float3 minbb, maxbb;
    float *coms;
    int nsolid;
    int *tags, *counts;

    EMALLOC(MAX_SOLIDS * 3 * 10, &coms);
    
    nsolid = read_coms(fname, /**/ coms);
    
    if (nsolid == 0) ERR("No solid provided.\n");

    mesh_get_bbox(rgi.vv, rgi.nv, /**/ &minbb, &maxbb);

    nsolid = duplicate_PBC(coords, rgi.pi, minbb, maxbb, nsolid, /**/ coms);
    make_local(coords, nsolid, /**/ coms);

    EMALLOC(*fluinfo.n, &tags);
    EMALLOC(nsolid, &counts);
    
    gen2(coords, comm, rgi, nsolid, coms, /**/ fluinfo, riginfo, /*w*/ tags, counts);

    EFREE(coms);
    EFREE(tags);
    EFREE(counts);
}

void gen_rig_from_solvent(const Coords *coords, MPI_Comm comm, RigGenInfo rgi,
                          /* io */ FluInfo fluinfo, /* o */ RigInfo riginfo) {
    // generate models
    msg_print("start rigid gen");
    gen(coords, comm, "rigs-ic.txt", rgi, /**/ fluinfo, riginfo);
    msg_print("done rigid gen");

    *riginfo.n = (*riginfo.ns) * (*riginfo.nps);
}
