void vel(MPI_Comm comm, const int *cc, int color, int n, /**/ Particle *pp) {
    dev::Map m;
    m.cc = cc; m.color = color;
    vel0(comm, m, n, /**/ pp);
}
