namespace ply {

void write(const char *fname, int nt, int nv, const int *tt, const float *vv) {
    int i;
    FILE *f;
    UC(efopen(fname, "w", &f));

    fprintf(f, "ply\n");
    fprintf(f, "format ascii 1.0\n");
    fprintf(f, "element vertex %d\n", nv);
    fprintf(f, "property float x\n");
    fprintf(f, "property float y\n");
    fprintf(f, "property float z\n");
    fprintf(f, "element face %d\n", nt);
    fprintf(f, "property list int int vertex_index\n");
    fprintf(f, "end_header\n");

    for (i = 0; i < nv; ++i)
        fprintf(f, "%g %g %g\n", vv[3*i + 0], vv[3*i + 1], vv[3*i + 2]);

    for (i = 0; i < nt; ++i)
        fprintf(f, "3 %d %d %d\n", tt[3*i + 0], tt[3*i + 1], tt[3*i + 2]);

    UC(efclose(f));
}

} // ply
