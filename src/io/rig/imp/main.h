#define BASE DUMP_BASE "/diag/rig"

static int swrite_v(const float v[3], char *buf) {
    enum {X, Y, Z};
    return sprintf(buf, "%+.6e %+.6e %+.6e ", v[X], v[Y], v[Z]); // 3 * 13 + 3 = 42
}

static int swrite(const Coords *c, int ns, const Solid *ss, int maxn, char *buf) {
    enum {X, Y, Z, D};
    int i, nchar;
    float com[D];
    const Solid *s;

    for (i = nchar = 0; i < ns; ++i) {
        s = &ss[i];

        // shift to global coordinates
        com[X] = xl2xg(c, s->com[X]);
        com[Y] = yl2yg(c, s->com[Y]);
        com[Z] = zl2zg(c, s->com[Z]);

        nchar += sprintf(buf + nchar, "%04d ", (int) s->id); // 5
        nchar += swrite_v(com,   buf + nchar);
        nchar += swrite_v(s->v,  buf + nchar);
        nchar += swrite_v(s->om, buf + nchar);
        nchar += swrite_v(s->fo, buf + nchar);
        nchar += swrite_v(s->to, buf + nchar);
        nchar += swrite_v(s->e0, buf + nchar);
        nchar += swrite_v(s->e1, buf + nchar);
        nchar += swrite_v(s->e2, buf + nchar);
        nchar += sprintf(buf + nchar, "\n"); // 1
        if (nchar > maxn)
            ERR("exceed buffer capacity: [%d / %d]", nchar, maxn);
    }
    return nchar;
}

static void write_mpi(MPI_Comm comm, const char *fname, long n, const char *data) {
    WriteFile *f;
    UC(write_file_open(comm, fname, &f));
    UC(write_all(comm, data, n, f));
    UC(write_file_close(f));
}

static void gen_fname(const char *name, long id, char *fname) {
    sprintf(fname, BASE "/%s.%04ld.txt", name, id);
}

void io_rig_dump(MPI_Comm comm, const Coords *c, const char *name, long id, int ns, const Solid *ss) {
    char fname[FILENAME_MAX], *data;
    int maxnc, nchar = 0;    

    enum {
        NC_ID = 5,
        NC_END = 1,
        NC_v3 = 42,
        NC_SECURITY = 1,
        N_v3 = 8
    };
    
    maxnc = ns * (NC_ID + NC_END + NC_v3 * N_v3 + NC_SECURITY);

    EMALLOC(maxnc, &data);

    if (m::is_master(comm))
        UC(os_mkdir(BASE));

    gen_fname(name, id, fname);
    nchar = swrite(c, ns, ss, maxnc, data);
    write_mpi(comm, fname, nchar, data);
    
    EFREE(data);
}

#undef BASE
