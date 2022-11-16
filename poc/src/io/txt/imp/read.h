#define VFRMT "%f %f %f"

static void ini(TxtRead **pq) {
    TxtRead *q;
    EMALLOC(1, &q);
    q->pp = NULL;
    q->ff = NULL;
    q->n  = -1;
    *pq = q;
}

static int get_num_lines(FILE *f) {
    int n;
    char c;
    n = 0;
    while (EOF != (c = fgetc(f)))
        if (c == '\n') ++n;
    return n;
}

void txt_read_pp(const char *name, TxtRead **pq) {
    enum {X, Y, Z};
    TxtRead *q;
    FILE *f;
    Particle p;
    int i;
    UC(ini(&q));
    q->ff = NULL;
    UC(efopen(name, "r", /**/ &f));
    q->n = get_num_lines(f);
    EMALLOC(q->n, &q->pp);
    rewind(f);
    i = 0;
    while (6 == fscanf(f, VFRMT " " VFRMT "\n",
                       p.r + X, p.r + Y, p.r + Z,
                       p.v + X, p.v + Y, p.v + Z)) {
        q->pp[i++] = p;
    }
    UC(efclose(f));
    *pq = q;
}

void txt_read_pp_ff(const char *name, TxtRead **pq) {
    enum {X, Y, Z};
    TxtRead *q;
    FILE *f;
    Particle p;
    Force fo;
    int i;
    UC(ini(&q));
    UC(efopen(name, "r", /**/ &f));
    q->n = get_num_lines(f);
    EMALLOC(q->n, &q->pp);
    EMALLOC(q->n, &q->ff);
    rewind(f);
    i = 0;
    while (9 == fscanf(f, VFRMT " " VFRMT " " VFRMT "\n",
                       p.r + X, p.r + Y, p.r + Z,
                       p.v + X, p.v + Y, p.v + Z,
                       fo.f + X, fo.f + Y, fo.f + Z)) {
        q->pp[i] = p;
        q->ff[i] = fo;
        ++i;
    }
    UC(efclose(f));
    *pq = q;    
}

void txt_read_ff(const char *name, TxtRead **pq) {
    enum {X, Y, Z};
    TxtRead *q;
    FILE *f;
    Force fo;
    int i;
    UC(ini(&q));
    UC(efopen(name, "r", /**/ &f));
    q->n = get_num_lines(f);
    EMALLOC(q->n, &q->ff);    
    rewind(f);
    i = 0;
    while (3 == fscanf(f, VFRMT "\n",
                       fo.f + X, fo.f + Y, fo.f + Z)) {
        q->ff[i] = fo;
        ++i;
    }
    UC(efclose(f));
    *pq = q;
}


void txt_read_fin(TxtRead *q) {
    EFREE(q->pp);
    EFREE(q->ff);
    EFREE(q);
}

int txt_read_get_n(const TxtRead *q) {return q->n;}
const Particle* txt_read_get_pp(const TxtRead *q) { return q->pp; }
const Force*    txt_read_get_ff(const TxtRead *q) { return q->ff; }

#undef VFRMT
