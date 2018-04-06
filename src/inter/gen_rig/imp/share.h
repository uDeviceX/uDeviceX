static void share0(MPI_Comm comm, const int root, /**/ Particle *pp, int *n) {
    int rank, size;
    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_size(comm, &size));
    std::vector<int> counts(size), displs(size);
    std::vector<Particle> recvbuf(MAX_PSOLID_NUM);
    MC(MPI_Gather(n, 1, MPI_INT, counts.data(), 1, MPI_INT, root, comm) );

    if (rank == root) {
        displs[0] = 0;
        for (int j = 0; j < size - 1; ++j)
            displs[j+1] = displs[j] + counts[j];
    }

    MC(m::Gatherv(pp, *n,
                  datatype::particle, recvbuf.data(), counts.data(), displs.data(),
                  datatype::particle, root, comm) );

    if (rank == root) {
        *n = displs.back() + counts.back();
        for (int i = 0; i < *n; ++i) pp[i] = recvbuf[i];
    }
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
