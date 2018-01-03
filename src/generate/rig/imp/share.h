static void share0(MPI_Comm comm, const int root, /**/ Particle *pp, int *n) {
    std::vector<int> counts(m::size), displs(m::size);
    std::vector<Particle> recvbuf(MAX_PSOLID_NUM);
    MC(MPI_Gather(n, 1, MPI_INT, counts.data(), 1, MPI_INT, root, comm) );

    if (m::rank == root)
    {
        displs[0] = 0;
        for (int j = 0; j < m::size - 1; ++j)
            displs[j+1] = displs[j] + counts[j];
    }

    MC(MPI_Gatherv(pp, *n,
                   datatype::particle, recvbuf.data(), counts.data(), displs.data(),
                   datatype::particle, root, comm) );

    if (m::rank == root) {
        *n = displs.back() + counts.back();
        for (int i = 0; i < *n; ++i) pp[i] = recvbuf[i];
    }
}

static void share(Coords coords, MPI_Comm comm, const int root, /**/ Particle *pp, int *n) {
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
