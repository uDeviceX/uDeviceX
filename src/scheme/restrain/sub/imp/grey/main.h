void vel(MPI_Comm comm, int n, /**/ Particle *pp) {
    dev::Map m;
    vel0(comm, m, n, /**/ pp);
}
