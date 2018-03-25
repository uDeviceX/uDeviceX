static void header(Out *o) {
    print(o, "# vtk DataFile Version 2.0\n");
    print(o, "created with uDeviceX\n");
    print(o, "BINARY\n");
    print(o, "DATASET POLYDATA\n");
}

static void points(Out *o, int n, double *rr) {
    MPI_Comm comm;
    WriteFile *file;
    print(o, "POINTS %d double\n", n);
    comm = o->comm;
    file = o->file;
    
    big_endian_dbl(3*n, /**/ rr);
    
    UC(write_all(comm, rr, 3*n*sizeof(rr[0]), file));
}
