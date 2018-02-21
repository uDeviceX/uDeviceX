enum {X, Y, Z};

void io_rig_ini(IoRig **iop) {
    IoRig *io;
    EMALLOC(1, iop);
    io = *iop;
    strcpy(io->mode, "w");
}

void io_rig_fin(IoRig *io) {
    EFREE(io);
}

static void write_v(FILE *f, const float v[3]) {
    fprintf(f, "%+.6e %+.6e %+.6e ", v[X], v[Y], v[Z]);
}

static void set_created(IoRig *io) {
    strcpy(io->mode, "a");
}

void io_rig_dump(const Coords *c, float t, int ns, const Solid *ss, const Solid *ssbb, IoRig *io) {
    enum {X, Y, Z};
    char fname[256];
    float com[3];
    FILE *fp;
    int j;
    const Solid *s, *sbb;

    for (j = 0; j < ns; ++j) {
        s   = ss   + j;
        sbb = ssbb + j;

        sprintf(fname, DUMP_BASE "/solid_diag_%04d.txt", (int) s->id);
        UC(efopen(fname, io->mode, /**/ &fp));
        fprintf(fp, "%+.6e ", t);

        // shift to global coordinates
        com[X] = xl2xg(c, s->com[X]);
        com[Y] = yl2yg(c, s->com[Y]);
        com[Z] = zl2zg(c, s->com[Z]);
            
        write_v(fp, com);
        write_v(fp, s->v );
        write_v(fp, s->om);
        write_v(fp, s->fo);
        write_v(fp, s->to);
        write_v(fp, s->e0);
        write_v(fp, s->e1);
        write_v(fp, s->e2);
        write_v(fp, sbb->fo);
        write_v(fp, sbb->to);
        fprintf(fp, "\n");
        
        UC(efclose(fp));
    }
    set_created(io);
}
