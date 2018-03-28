#define PATTERN "%s/%s/%05d.ply"
enum { NVP = 3 /* number of vertices per face */ };

static void ini(MPI_Comm comm, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    int i;
    MeshWrite *q;
    EMALLOC(1, &q);

    q->nv = nv; q->nt = nt;
    cpy(q->directory, directory);
    UC(mkdir(comm, DUMP_BASE, directory));    
    EMALLOC(nt, &q->tt);
    for (i = 0; i < nt; i++)
        q->tt[i] = tt[i];
    q->shift_type = get_shift_type();
    *pq = q;
}

void mesh_write_ini(MPI_Comm comm, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    UC(ini(comm, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_ini_off(MPI_Comm comm, MeshRead *cell, const char *directory, /**/ MeshWrite **pq) {
    int nv, nt;
    const int4 *tt;
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tt = mesh_read_get_tri(cell);
    UC(ini(comm, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_fin(MeshWrite *q) {
    EFREE(q->tt);
    EFREE(q);
}

static void header(MPI_Comm comm, int nc0, int nv, int nt, WriteFile *f) {
    int nc; /* total number of cells */
    int sz = 0;
    char s[BUFSIZ];
    UC(write_reduce(comm, nc0, &nc));
    if (m::is_master(comm))
        sz = snprintf(s, BUFSIZ - 1,
                     "ply\n"
                     "format binary_little_endian 1.0\n"
                     "element vertex %d\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property float u\nproperty float v\nproperty float w\n"
                     "element face  %d\n"
                     "property list int int vertex_index\n"
                     "end_header\n", nv*nc, nt*nc);
    write_master(comm, s, sz, f);
}

static void vert(MPI_Comm cart, int n, const Vectors *pos, const Vectors *vel, WriteFile *f) {
    float *D, *r, *v;
    int i, j, m;
    m = (3 + 3) * n; /* pos[3] + vel[3] */
    EMALLOC(m, &D);
    for (i = j = 0; i < n; i++) {
        r = &D[j]; j += 3;
        v = &D[j]; j += 3;
        UC(vectors_get(pos, i, /**/ r));
        UC(vectors_get(vel, i, /**/ v));
    }
    UC(write_all(cart, D, m*sizeof(D[0]), f));
    EFREE(D);
}

static void wfaces0(MPI_Comm comm, int *buf, const int4 *faces, int nc, int nv, int nt, WriteFile *f) {
    /* write faces */
    int c, t, b;  /* cell, triangle, buffer index */
    int n, shift;
    int4 tri;
    n = nc * nv;
    UC(write_shift_indices(comm, n, &shift));
    for (b = c = 0; c < nc; ++c)
        for (t = 0; t < nt; ++t) {
            tri = faces[t];
            buf[b++] = NVP;
            buf[b++] = shift + nv*c + tri.x;
            buf[b++] = shift + nv*c + tri.y;
            buf[b++] = shift + nv*c + tri.z;
        }
    UC(write_all(comm, buf, b*sizeof(buf[0]), f));
}

static void wfaces(MPI_Comm cart, const int4 *faces, int nc, int nv, int nt, WriteFile *f) {
    int *buf; /* buffer for faces */
    int sz;
    sz = (1 + NVP) * nc * nt;
    EMALLOC(sz, &buf);
    UC(wfaces0(cart, buf, faces, nc, nv, nt, f));
    EFREE(buf);
}

static void mesh_write(MPI_Comm comm, int nc, int nv, int nt,
                        const Vectors *pos, const Vectors *vel, const int4 *faces,
                        WriteFile *f) {
    int n;
    n = nc * nv;
    UC(header(comm, nc, nv, nt, f));
    UC(vert(comm, n, pos, vel, f));
    UC(wfaces(comm, faces, nc, nv, nt, f));
}

void mesh_write_particles(MeshWrite *q, MPI_Comm comm, const Coords *coords, int nc, const Particle *pp, int id) {
    int nv, nt, n;
    const int4 *tt;
    Vectors *pos, *vel;
    WriteFile *f;
    const char *directory;
    char path[FILENAME_MAX];

    nv = q->nv; nt = q->nt; tt = q->tt; directory = q->directory;
    n = nv * nc;
    if (sprintf(path, PATTERN, DUMP_BASE, directory, id) < 0)
        ERR("sprintf failed");
    UC(write_file_open(comm, path, /**/ &f));
    switch (q->shift_type) {
    case EDGE:
        vectors_postions_edge_ini(coords, n, pp, /**/ &pos);
        break;
    case CENTER:
        vectors_postions_center_ini(coords, n, pp, /**/ &pos);
        break;
    default:
        ERR("unkown q->type: %d", q->shift_type);
    }
    vectors_velocities_ini(n, pp, /**/ &vel);
    UC(mesh_write(comm, nc, nv, nt, pos, vel, tt, f));
    vectors_fin(pos);
    vectors_fin(vel);
    UC(write_file_close(f));
}

void mesh_write_vectros(MeshWrite *q, MPI_Comm comm, int nc, Vectors *pos, Vectors *vel, int id) {
    int nv, nt;
    const int4 *tt;
    WriteFile *f;
    const char *directory;
    char path[FILENAME_MAX];

    nv = q->nv; nt = q->nt; tt = q->tt; directory = q->directory;
    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, directory, id) < 0)
        ERR("snprintf failed");
    UC(write_file_open(comm, path, /**/ &f));
    UC(mesh_write(comm, nc, nv, nt, pos, vel, tt, f));
    UC(write_file_close(f));
}
