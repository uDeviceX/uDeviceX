static void header(Out *o) {
    print(o, "# vtk DataFile Version 2.0\n");
    print(o, "created with uDeviceX\n");
    print(o, "BINARY\n");
    print(o, "DATASET POLYDATA\n");
}

static void points(Out *o, int n, double *rr) {
    int n_total;
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    big_endian_dbl(3*n, /**/ rr);
    UC(write_reduce(comm, n, &n_total));
    print(o, "POINTS %d double\n", n_total);
    UC(write_all(comm, rr, 3*n*sizeof(rr[0]), file));
    print(o, "\n");
}

static void tri(Out *o, int nm, int nv, int nt, const int *tt, int *buf) {
    enum {NVP = 3};
    int n, m, t, i, shift, nm_total;
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    n = nm * nv;
    UC(write_shift_indices(comm, n, &shift));
    for (t = m = 0; m < nm; m++)
        for (i = 0; i < nt; i++) {
            buf[t++] = NVP;
            buf[t++] = shift + nv*m + tt[3*i + 0];
            buf[t++] = shift + nv*m + tt[3*i + 1];
            buf[t++] = shift + nv*m + tt[3*i + 2];
        }
    big_endian_int(t, /**/ buf);
    UC(write_reduce(comm, nm, &nm_total));
    print(o, "POLYGONS %d %d\n", nt*nm_total, 4*nt*nm_total);
    UC(write_all(comm, buf, t*sizeof(buf[0]), file));
    print(o, "\n");
}

static void cell_header(Out *o, int n) {
    int n_total;
    MPI_Comm comm;
    comm = o->comm;
    UC(write_reduce(comm, n, &n_total));
    print(o, "CELL_DATA %d\n", n_total);
}

static void cell_data(Out *o, int n, double *data, const char *name) {
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    print(o, "SCALARS %s double 1\n", name);
    print(o, "LOOKUP_TABLE default\n");
    
    big_endian_dbl(n, /**/ data);
    UC(write_all(comm, data, n*sizeof(data[0]), file));
    print(o, "\n");
}

static void point_header(Out *o, int n) {
    int n_total;
    MPI_Comm comm;
    comm = o->comm;
    UC(write_reduce(comm, n, &n_total));
    print(o, "POINT_DATA %d\n", n_total);
}

static void point_data(Out *o, int n, double *data, const char *name) {
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    print(o, "SCALARS %s double 1\n", name);
    print(o, "LOOKUP_TABLE default\n");
    
    big_endian_dbl(n, /**/ data);
    UC(write_all(comm, data, n*sizeof(data[0]), file));
    print(o, "\n");
}
