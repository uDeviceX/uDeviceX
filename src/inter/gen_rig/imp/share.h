static void share0(MPI_Comm comm, const int root, /**/ Particle *pp, int *n) {
    int rank, size, i;
    int *counts, *displs;
    Particle *recvbuf;
    MPI_Datatype PartType;

    MC(MPI_Type_contiguous(sizeof(Particle) / sizeof(float), MPI_FLOAT, &PartType));
    MC(MPI_Type_commit(&PartType));

    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_size(comm, &size));

    EMALLOC(size, &counts);
    EMALLOC(size, &displs);
    EMALLOC(MAX_PSOLID_NUM, &recvbuf);

    MC(m::Gather(n, 1, MPI_INT, counts, 1, MPI_INT, root, comm) );

    if (rank == root) {
        displs[0] = 0;
        for (i = 0; i < size - 1; ++i) displs[i+1] = displs[i] + counts[i];
    }

    MC(m::Gatherv(pp, *n, PartType, recvbuf, counts, displs, PartType, root, comm) );

    if (rank == root) {
        *n = displs[size-1] + counts[size-1];
        for (i = 0; i < *n; ++i) pp[i] = recvbuf[i];
    }

    MC(MPI_Type_free(&PartType));
}

static void share(const Coords *coords, MPI_Comm comm, const int root, /**/ Particle *pp, int *n) {
    if (*n >= MAX_PSOLID_NUM) ERR("Number of solid particles too high for the buffer\n");
    // set to global coordinates and then convert back to local
    int i;
    Particle p;
    enum {X, Y, Z};

    for (i = 0; i < *n; ++i) {
        p = pp[i];
        p.r[X] = xl2xg(coords, p.r[X]);
        p.r[Y] = yl2yg(coords, p.r[Y]);
        p.r[Z] = zl2zg(coords, p.r[Z]);
    }
    
    share0(comm, root, /**/ pp, n);

    for (i = 0; i < *n; ++i) {
        p = pp[i];
        p.r[X] = xg2xl(coords, p.r[X]);
        p.r[Y] = yg2yl(coords, p.r[Y]);
        p.r[Z] = zg2zl(coords, p.r[Z]);
    }    
}
