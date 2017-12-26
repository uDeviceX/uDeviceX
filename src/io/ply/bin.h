namespace ply {
void write(const char *fname, int nt, int nv, const int *tt, const float *vv) {
    FILE *f;

    UC(efopen(fname, "wb", /**/ &f));
    {
        char header[1024];
        sprintf(header,
                "ply\n"
                "format binary_little_endian 1.0\n"
                "element vertex %d\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "element face %d\n"
                "property list int int vertex_index\n"
                "end_header\n",
                nv, nt);

        UC(efwrite(header, sizeof(char), strlen(header), f));
    }

    UC(efwrite(vv, sizeof(float), 3 * nv, f));

    int *ibuf = new int[4 * nt];

    for (int i = 0; i < nt; ++i)
        {
            ibuf[4*i + 0] = 3;
            ibuf[4*i + 1] = tt[3*i + 0];
            ibuf[4*i + 2] = tt[3*i + 1];
            ibuf[4*i + 3] = tt[3*i + 2];
        }

    UC(efwrite(ibuf, sizeof(int), 4 * nt, f));

    delete[] ibuf;

    UC(efclose(f));
}
} // ply
