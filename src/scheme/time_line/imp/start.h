#define CODE "timeline"

static void write_state(const TimeLine *t, FILE *f) {
    fprintf(f, "%g %g\n", t->t,  t->t0);
    fprintf(f, "%g %g\n", t->dt, t->dt0);
    fprintf(f, "%ld\n", t->iteration);
}

static void stream_write_state(const void *data, FILE *f) {
    const TimeLine *t = (const TimeLine*) data;
    write_state(t, f);
}

void time_line_strt_dump(MPI_Comm comm, const char *base, int id, const TimeLine *t) {
    StreamWriter sw = stream_write_state;
    restart_write_stream_one_node(comm, base, CODE, id, (void*) t, sw);
}

static void read_state(FILE *f, TimeLine *t) {
    fscanf(f, "%g %g\n", &t->t,  &t->t0);
    fscanf(f, "%g %g\n", &t->dt, &t->dt0);
    fscanf(f, "%ld\n", &t->iteration);
}

static void stream_read_state(FILE *f, void *data) {
    TimeLine *t = (TimeLine*) data;
    UC(read_state(f, t));
}

void time_line_strt_read(const char *base, int id, TimeLine *t) {
    StreamReader sr = stream_read_state;
    restart_read_stream_one_node(base, CODE, id, sr, (void*) t);
}

#undef CODE
