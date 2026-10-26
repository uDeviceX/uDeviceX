// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/scheme/time_line/imp.cpp */
#include <mpi.h>
#include <stdio.h>
#include <math.h>

/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
/* begin src/utils/error.h */
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)

/* udx check */
#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

/* [B]o[p] [c]heck */
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
/* end src/utils/error.h */
/* begin src/utils/imp.h */
// tag::mem[]
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>
// end::mem[]

// tag::macros[]
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))
// end::macros[]

// tag::stdio[]
void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);
// end::stdio[]

// tag::tools[]
bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
// end::tools[]
/* end src/utils/imp.h */

/* begin src/io/restart/imp.h */
struct Coords;
struct Particle;
struct Rigid;

// tag::id[]
#ifndef UDX_SIM_RESTART_COMMON
#define UDX_SIM_RESTART_COMMON
enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};
#endif
// end::id[]

// tag::stream[]
typedef void (*StreamWriter)(const void*, FILE*);
typedef void (*StreamReader)(FILE*, void*);
// end::stream[]

// tag::write[]
void restart_write_pp(MPI_Comm, const char *base, const char *code, int id, long n, const Particle *pp); // <1>
void restart_write_ii(MPI_Comm, const char *base, const char *code, int id, long n, const int *ii);      // <2>
void restart_write_ss(MPI_Comm, const char *base, const char *code, int id, long n, const Rigid *ss);    // <3>
void restart_write_stream_one_node(MPI_Comm, const char *base, const char *code, int id, const void *data, StreamWriter); // <4>
 // end::write[]

// tag::read[]
void restart_read_pp(MPI_Comm, const char *base, const char *code, int id, int *n, Particle *pp);
void restart_read_ii(MPI_Comm, const char *base, const char *code, int id, int *n, int *ii);
void restart_read_ss(MPI_Comm, const char *base, const char *code, int id, int *n, Rigid *ss);
void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader, void *data);
// end::read[]
/* end src/io/restart/imp.h */

/* begin src/scheme/time_line/imp.h */
struct TimeLine;

// tag::mem[]
void time_line_ini(float start, /**/ TimeLine**);
void time_line_fin(TimeLine*);
// end::mem[]

// tag::int[]
void time_line_advance(float dt, TimeLine*);           // <1>
int  time_line_cross(const TimeLine*, float interval); // <2>
// end::int[]

// tag::strt[]
void time_line_strt_dump(MPI_Comm, const char *base, int id, const TimeLine*); // <1>
void time_line_strt_read(const char *base, int id, TimeLine*); // <2>
// end::strt[]

// tag::get[]
float time_line_get_current  (const TimeLine*); // <1>
long  time_line_get_iteration(const TimeLine*); // <2>
// end::get[]
/* end src/scheme/time_line/imp.h */

/* begin src/scheme/time_line/imp/type.h */
struct TimeLine {
    float t, t0;
    float dt, dt0;
    long iteration;
};
/* end src/scheme/time_line/imp/type.h */
/* begin src/scheme/time_line/imp/main.h */
void time_line_ini(float t, /**/ TimeLine** pq) {
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
long  time_line_get_iteration(const TimeLine *q) { return q->iteration; }
int time_line_cross(const TimeLine *q, float i) {
    float t, t0, f;
    if (q->iteration == 0) return 0;
    else if (i < 0)   ERR("i < 0");
    else if (i == 0)  return 1;

    t = q->t; t0 = q->t0;
    f = floor(t / i) * i;
    return t0 <= f && f < t;
}
/* end src/scheme/time_line/imp/main.h */
/* begin src/scheme/time_line/imp/start.h */
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
/* end src/scheme/time_line/imp/start.h */
/* end src/scheme/time_line/imp.cpp */

#line 1 "src/scheme/time_step/imp.cpp"
