void set_rig_ids(const int n, Solid *ss) {
    int j, id = 0;
    MC(MPI_Exscan(&n, &id, 1, MPI_INT, MPI_SUM, m::cart));

    for (j = 0; j < n; ++j)
        ss[j].id = id++;
}
