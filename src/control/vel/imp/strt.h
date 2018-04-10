#define PAT "%g %g %g\n"

static void writef3(float3 v, FILE *f) {
    fprintf(f, PAT, v.x, v.y, v.z);
}

static void write_state(const State *s, FILE *f) {
    writef3(s->cur, f);
    writef3(s->olde, f);
    writef3(s->sume, f);
    writef3(s->f, f);
}

void vcont_stream_write_state(const PidVCont *c, FILE *f) {
    write_state(&c->state, f);
}

static void readf3(FILE *f, float3 *v) {
    if (3 != fscanf(f, PAT, &v->x, &v->y, &v->z))
        ERR("wrong format\n");
}

static void read_state(FILE *f, State *s) {
    UC(readf3(f, &s->cur));
    UC(readf3(f, &s->olde));
    UC(readf3(f, &s->sume));
    UC(readf3(f, &s->f));
}

void vcont_stream_read_state(FILE *f, PidVCont *c) {
    UC(read_state(f, &c->state));
}

#undef PAT
