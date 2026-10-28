#define DEV_CUDA
#define CC_RELEASE

#include "config.cuh"
#include "macros.cuh"
#include "types.cuh"
#include "constants.cuh"

#include <stdio.h>

void emalloc(size_t, void **);
void efree(void *);
void efopen(const char *, const char *, FILE **);
void efclose(FILE *);
void msg_print(const char *, ...);

void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);

struct TxtRead {
    int n;
    Particle *pp;
    Force *ff;
};
#define VFRMT "%16.10e %16.10e %16.10e"

void txt_write_pp(long n, const Particle *pp, const char *name) {
    enum {X, Y, Z};
    long i;
    FILE *f;
    const float *r, *v;
    int rcode;

    UC(efopen(name, "w",      &f));
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        v = pp[i].v;
        rcode = fprintf(f, VFRMT " " VFRMT "\n",
                        r[X], r[Y], r[Z], v[X], v[Y], v[Z]);
        if (rcode < 0) ERR("fprintf to file '%s' failed", name);
    }
    UC(efclose(f));
    msg_print("dump %ld particles to '%s'", n, name);
}

void txt_write_pp_ff(long n, const Particle *pp, const Force *ff, const char *name) {
    enum {X, Y, Z};
    long i;
    FILE *f;
    const float *r, *v, *a;
    int rcode;

    UC(efopen(name, "w",      &f));
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        v = pp[i].v;
        a = ff[i].f;
        rcode = fprintf(f, VFRMT " " VFRMT " " VFRMT "\n",
                        r[X], r[Y], r[Z], v[X], v[Y], v[Z], a[X], a[Y], a[Z]);
        if (rcode < 0) ERR("fprintf to file '%s' failed", name);
    }
    UC(efclose(f));
    msg_print("dump %ld particles to '%s'", n, name);
}

#undef VFRMT
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
    UC(efopen(name, "r",      &f));
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
    UC(efopen(name, "r",      &f));
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
    UC(efopen(name, "r",      &f));
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

