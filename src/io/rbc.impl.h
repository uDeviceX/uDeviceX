static void write(const void * const ptr, const int nbytes32, MPI_File f) {
    MPI_Offset base;
    MC(MPI_File_get_position(f, &base));
    MPI_Offset offset = 0, nbytes = nbytes32;
    MC(MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart));
    MPI_Status status;
    MC(MPI_File_write_at_all(f, base + offset, ptr, nbytes, MPI_CHAR, &status));
    MPI_Offset ntotal = 0;
    MC(MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, m::cart) );
    MC(MPI_File_seek(f, ntotal, MPI_SEEK_CUR));
}

static void rbc_dump0(const char * filename,
              int *mesh_indices, const int ninstances, const int ntriangles_per_instance,
              Particle * _particles, int nvertices_per_instance) {
    std::vector<Particle> particles(_particles, _particles + ninstances * nvertices_per_instance);
    int NPOINTS = 0;
    const int n = particles.size();
    l::m::Reduce(&n, &NPOINTS, 1, MPI_INT, MPI_SUM, 0, m::cart) ;
    const int ntriangles = ntriangles_per_instance * ninstances;
    int NTRIANGLES = 0;
    l::m::Reduce(&ntriangles, &NTRIANGLES, 1, MPI_INT, MPI_SUM, 0, m::cart) ;
    MPI_File f;
    MPI_File_open(m::cart, filename,
                  MPI_MODE_WRONLY |  MPI_MODE_CREATE,
                  MPI_INFO_NULL, &f);
    MPI_File_set_size (f, 0);

    std::stringstream ss;
    if (m::rank == 0) {
        ss <<  "ply\n";
        ss <<  "format binary_little_endian 1.0\n";
        ss <<  "element vertex " << NPOINTS << "\n";
        ss <<  "property float x\nproperty float y\nproperty float z\n";
        ss <<  "property float u\nproperty float v\nproperty float w\n";
        ss <<  "element face " << NTRIANGLES << "\n";
        ss <<  "property list int int vertex_index\n";
        ss <<  "end_header\n";
    }
    std::string content = ss.str();
    write(content.c_str(), content.size(), f);
    int L[3] = { XS, YS, ZS };
    for(int i = 0; i < n; ++i)
    for(int c = 0; c < 3; ++c)
    particles[i].r[c] += L[c] / 2 + m::coords[c] * L[c];

    write(&particles.front(), sizeof(Particle) * n, f);
    int poffset = 0;
    MPI_Exscan(&n, &poffset, 1, MPI_INTEGER, MPI_SUM, m::cart);
    std::vector<int> buf;
    for(int j = 0; j < ninstances; ++j)
    for(int i = 0; i < ntriangles_per_instance; ++i) {
        int primitive[4] = { 3,
                             poffset + nvertices_per_instance * j + mesh_indices[3 * i + 0],
                             poffset + nvertices_per_instance * j + mesh_indices[3 * i + 1],
                             poffset + nvertices_per_instance * j + mesh_indices[3 * i + 2] };
        buf.insert(buf.end(), primitive, primitive + 4);
    }
    write(&buf.front(), sizeof(int) * buf.size(), f);
    MPI_File_close(&f);
}

void rbc_dump(Particle *pp, int *faces, int nc, int nv, int nt, int id) {
    const char *fmt = DUMP_BASE "/r/%05d.ply";
    char buf[BUFSIZ];
    sprintf(buf, fmt, id);
    if (m::rank == 0) os::mkdir(DUMP_BASE "/r");
    rbc_dump0(buf, faces, nc, nt, pp, nv);
}
