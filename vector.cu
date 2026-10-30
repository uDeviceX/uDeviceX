#include "memory.cuh"
#include "macros.cuh"
#include "types.cuh"
#include <stdio.h>

struct Coords;
struct Tform { float o[3], s[3]; };
void error_stack_push(const char *file, int line);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *file, int line, const char *fmt, ...);


float xl2xg(const Coords*, float);
float yl2yg(const Coords*, float);
float zl2zg(const Coords*, float);
float xl2xc(const Coords*, float);
float yl2yc(const Coords*, float);
float zl2zc(const Coords*, float);
void tform_ini(Tform**);
void tform_fin(Tform*);
void tform_vector(const float[3], const float[3], const float[3], const float[3], Tform*);
void tform_convert(const Tform*, const float[3], float[3]);

struct Vectors {
    int type;
    int n;
    union {
        const float *rr;
        const Particle *pp;
    } D;
    Tform *tform;
};

static void float_get(const Vectors*, int i, float r[3]);

static void positions_get(const Vectors*, int i, float r[3]);
static void positions_edge_get(const Vectors*, int i, float r[3]);
static void positions_center_get(const Vectors*, int i, float r[3]);
static void velocities_get(const Vectors*, int i, float r[3]);
static void zero_get(const Vectors*, int i, float r[3]);

enum {VECTOR_FLOAT, VECTOR_POSITIONS, VECTOR_POSITIONS_EDGE, VECTOR_POSITIONS_CENTER, VECTOR_VELOCITIES, VECTOR_ZERO};
static void (*vector_get_table[])(const Vectors*, int i, float r[3]) =
{ float_get,  positions_get, positions_edge_get, positions_center_get, velocities_get, zero_get };

typedef  void (*Local2Global)(const Coords *coords, float a[3],      float b[3]);
void vectors_float_ini(int n, const float *rr,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_FLOAT; q->n = n; q->D.rr = rr;
    *pq = q;
}

void vectors_postions_ini(int n, const Particle *pp,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_POSITIONS; q->n = n; q->D.pp = pp;
    *pq = q;
}

static void l2edge(const Coords *coords, float a[3],      float b[3]) {
    enum {X, Y, Z};
    b[X] = xl2xg(coords, a[X]);
    b[Y] = yl2yg(coords, a[Y]);
    b[Z] = zl2zg(coords, a[Z]);
}
static void l2center(const Coords *coords, float a[3],      float b[3]) {
    enum {X, Y, Z};
    b[X] = xl2xc(coords, a[X]);
    b[Y] = yl2yc(coords, a[Y]);
    b[Z] = zl2zc(coords, a[Z]);
}

static void edge_ini_tform(const Coords *coords, Local2Global local2global, Tform **pq) {
    enum {X, Y, Z};
    Tform *q;
    float a0[3], a1[3], b0[3], b1[3];
    a0[X] = a0[Y] = a0[Z] = 0;
    b0[X] = b0[Y] = b0[Z] = 1;
    local2global(coords, a0,      a1);
    local2global(coords, b0,      b1);
    tform_ini(&q);
    UC(tform_vector(a0, a1, b0, b1,      q));
    *pq = q;
}
void vectors_postions_edge_ini(const Coords *coords, int n, const Particle *pp,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_POSITIONS_EDGE; q->n = n; q->D.pp = pp;
    edge_ini_tform(coords, l2edge,      &q->tform);
    *pq = q;
}

void vectors_postions_center_ini(const Coords *coords, int n, const Particle *pp,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_POSITIONS_EDGE; q->n = n; q->D.pp = pp;
    edge_ini_tform(coords, l2center,      &q->tform);
    *pq = q;
}

void vectors_velocities_ini(int n, const Particle *pp,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_VELOCITIES; q->n = n; q->D.pp = pp;
    *pq = q;
}

void vectors_zero_ini(int n,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_ZERO; q->n = n;
    *pq = q;
}

void vectors_fin(Vectors *q) {
    if (q->type == VECTOR_POSITIONS_EDGE ||
        q->type == VECTOR_POSITIONS_CENTER)
        tform_fin(q->tform);
    EFREE(q);
}

static void float_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const float *rr;
    rr = q->D.rr;
    r[X] = rr[3*i + 0];
    r[Y] = rr[3*i + 1];
    r[Z] = rr[3*i + 2];
}
static void positions_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}

static void shift_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    UC(tform_convert(q->tform, pp[i].r,      r));
}
static void positions_edge_get(const Vectors *q, int i, float r[3]) { shift_get(q, i, r); }
static void positions_center_get(const Vectors *q, int i, float r[3]) { shift_get(q, i, r); }
static void velocities_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].v[X];
    r[Y] = pp[i].v[Y];
    r[Z] = pp[i].v[Z];
}
static void zero_get(const Vectors*, int, float r[3]) {
    enum {X, Y, Z};
    r[X] = r[Y] = r[Z] = 0;
}
void vectors_get(const Vectors *q, int i,      float r[3]) {
    int n;
    n = q->n;
    if (i >= n) ERR("i = %d    >=   n = %d", i, n);
    if (i < 0)  ERR("i = %d    < 0", i);
    vector_get_table[q->type](q, i, r);
}
