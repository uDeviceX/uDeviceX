static void header(Out *o) {
    print(o, "# vtk DataFile Version 2.0\n");
    print(o, "created with uDeviceX\n");
    print(o, "BINARY\n");
    print(o, "DATASET POLYDATA\n");
}

static void points(Out *o, int n, double *rr) {
    print(o, "POINTS %d double\n", n);
}
