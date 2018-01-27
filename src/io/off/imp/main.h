static int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

static void ini(OffRead **pq) {
    OffRead *p;
    UC(emalloc(sizeof(OffRead), (void**)&p));
    *pq = p;
}

#define SIZE 1024
static int emptyp(char *s) {
    for (;;) {
        if      (s[0] == '\0')                return 1;
        else if (s[0] != ' ' && s[0] != '\t') return 0;
        else s++;
    }
}
static int commentp(char *s) {
    return s[0] != '\0' && s[0] == '#';
}
static void unend(char *s) { /* rm \n */
    if (s[0] == '\0') return;
    while (s[1] != '\0') s++;
    if (s[0] == '\n') s[0] = '\0';
}
enum {OK, ERR};
static int line(FILE *f, /**/ char *s) { /* read line */
    for (;;) {
        if (fgets(s, SIZE, f) == NULL)  return ERR;
        unend(s);
        if (!emptyp(s) && !commentp(s)) break;
    }
    return OK;
}

static int header(FILE *f, char *s) {
    char key[SIZE];
    if (line(f, /**/ s) == ERR) return 0;
    sscanf(s, "%s", key);
    return eq(key, "OFF");
}

static int sizes(FILE *f, char *s, /**/ int *nv, int *nt) {
    int r;
    if (line(f, /**/ s) == ERR) return 0;
    r = sscanf(s, "%d %d %*d", nv, nt);
    return r == 2;
}
static void read(FILE *f, const char *path, /**/ OffRead *q) {
    int nv, nt;
    char s[SIZE];
    if (!header(f, /**/ s)) {
        msg_print("not an OFF file: '%s'", s);
        ERR("faild to read: '%s'", path);
    }
    if (!sizes(f, /**/ s, &nv, &nt)) {
        msg_print("wrong sizes: '%s'", s);
        ERR("faild to read: '%s'", path);
    }
    /* vert(f);
    tri(f); */
    q->nv = nv; q->nt = nt;
}

void off_read(const char *path, OffRead **pq) {
    FILE *f;
    OffRead *q;
    UC(ini(&q));
    UC(efopen(path, "r", /**/ &f));
    read(f, path, /**/ q);
    UC(efclose(f));
    *pq = q;
}

void off_fin(OffRead* q) {
    //    UC(efree(q->rr));
    //    UC(efree(q->tt));
    UC(efree(q));
}

int    off_get_n(OffRead*) {
    return 0;
}

int4  *off_get_tri(OffRead*) {
    int4 *q = NULL;
    return q;
}

float *off_get_vert(OffRead*) {
    float *q = NULL;
    return q;
}
