// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/io/txt/imp.cpp */
#include <stdio.h>

/* begin src/inc/type.h */
template <typename T, int N>
struct Sarray { /* [s]tatic array */
    T d[N];
};

struct Particle {
    float r[3], v[3];
};

struct Rigid {
    float Iinv[6],            /* moment of inertia            6        */
        mass,                 /* mass of the solid            7        */
        com[3],               /* [c]enter [o]f [m]ass         10       */
        v[3], om[3],          /* linear and angular velocity  13 16    */
        e0[3], e1[3], e2[3],  /* local referential            19 22 25 */
        fo[3], to[3],         /* force, torque                28 31    */
        id;                   /* id of the solid              32       */
};

struct Force {
    float f[3];
};

struct Momentum {
    float P[3], L[3]; /* linear and angular momentum */
};

struct Positioncp {
    float rc[3]; /* current position  */
    float rp[3]; /* previous position */
};

typedef Sarray<int, 26>       int26;
typedef Sarray<int, 27>       int27;

typedef Sarray<int*, 26>     intp26;
typedef Sarray<Particle*, 26> Pap26;
typedef Sarray<Force*,    26> Fop26;
typedef Sarray<Momentum*, 26> Mop26;
typedef Sarray<Positioncp*, 26> Pocpp26;

/* particles wrap */
struct PaWrap {
    int n;
    const Particle *pp;
};

/* forces wrap */
struct FoWrap {
    int n;
    Force *ff;
};
/* end src/inc/type.h */

/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
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

/* begin src/io/txt/imp.h */
struct Particle;
struct Force;
struct TxtRead;

// tag::write[]
void txt_write_pp(long n, const Particle*, const char *path);
void txt_write_pp_ff(long n, const Particle*, const Force*, const char *path);
// end::write[]

// tag::read[]
void txt_read_pp(const char *path, TxtRead **);
void txt_read_pp_ff(const char *path, TxtRead **);
void txt_read_ff(const char *path, TxtRead **);
void txt_read_fin(TxtRead*);

int txt_read_get_n(const TxtRead *);
const Particle* txt_read_get_pp(const TxtRead *);
const Force*    txt_read_get_ff(const TxtRead *);
// end::read[]
/* end src/io/txt/imp.h */

/* begin src/io/txt/imp/type.h */
struct TxtRead {
    int n;
    Particle *pp;
    Force *ff;
};
/* end src/io/txt/imp/type.h */
/* begin src/io/txt/imp/dump.h */
#define VFRMT "%16.10e %16.10e %16.10e"

void txt_write_pp(long n, const Particle *pp, const char *name) {
    enum {X, Y, Z};
    long i;
    FILE *f;
    const float *r, *v;
    int rcode;

    UC(efopen(name, "w", /**/ &f));
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

    UC(efopen(name, "w", /**/ &f));
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
/* end src/io/txt/imp/dump.h */
/* begin src/io/txt/imp/read.h */
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
/* end src/io/txt/imp/read.h */
/* end src/io/txt/imp.cpp */

#line 1 "src/io/vtk/imp.cpp"
