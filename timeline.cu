#include "memory.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include "config.cuh"
#include "macros.cuh"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);

typedef void (*StreamWriter)(const void *, FILE *);
typedef void (*StreamReader)(FILE *, void *);
void restart_write_stream_one_node(MPI_Comm, const char *, const char *, int, const void *, StreamWriter);
void restart_read_stream_one_node(const char *, const char *, int, StreamReader, void *);

struct TimeLine {
    float t, t0;
    float dt, dt0;
    long iteration;
};

void time_line_ini(float t, TimeLine **pq) {
    TimeLine *q;
    EMALLOC(1, &q);
    q->t = q->t0 = t;
    q->iteration = 0;
    *pq = q;
}

void time_line_fin(TimeLine *q) { EFREE(q); }

void time_line_advance(float dt, TimeLine *q) {
    if (dt < 0)
        ERR("dt = %g < 0", dt);
    if (q->iteration == 0) q->dt0 = dt;
    else                   q->t0 = q->t;

    q->t  += dt;
    q->dt0 = q->dt;
    q->dt  = dt;
    q->iteration++;
}

float time_line_get_current(const TimeLine *q) { return q->t; }
long time_line_get_iteration(const TimeLine *q) { return q->iteration; }

int time_line_cross(const TimeLine *q, float i) {
    float t, t0, f;
    if (q->iteration == 0) return 0;
    if (i < 0) ERR("i < 0");
    if (i == 0) return 1;

    t = q->t;
    t0 = q->t0;
    f = floor(t / i) * i;
    return t0 <= f && f < t;
}

static void write_state(const TimeLine *t, FILE *f) {
    fprintf(f, "%g %g\n", t->t, t->t0);
    fprintf(f, "%g %g\n", t->dt, t->dt0);
    fprintf(f, "%ld\n", t->iteration);
}

static void stream_write_state(const void *data, FILE *f) {
    write_state((const TimeLine *)data, f);
}

void time_line_strt_dump(MPI_Comm comm, const char *base, int id, const TimeLine *t) {
    restart_write_stream_one_node(comm, base, "timeline", id, t, stream_write_state);
}

static void read_state(FILE *f, TimeLine *t) {
    fscanf(f, "%g %g\n", &t->t, &t->t0);
    fscanf(f, "%g %g\n", &t->dt, &t->dt0);
    fscanf(f, "%ld\n", &t->iteration);
}

static void stream_read_state(FILE *f, void *data) {
    UC(read_state(f, (TimeLine *)data));
}

void time_line_strt_read(const char *base, int id, TimeLine *t) {
    restart_read_stream_one_node(base, "timeline", id, stream_read_state, t);
}
