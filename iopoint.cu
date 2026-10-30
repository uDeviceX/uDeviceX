#include "memory.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include "config.cuh"
#include "macros.cuh"
#include "constants.cuh"

#include <ctype.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

void os_mkdir(const char *);
void msg_print(const char *, ...);
bool same_str(const char *, const char *);
int string_nword(const char *);

void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);

struct BopData;

BopStatus bop_ini(BopData **);
BopStatus bop_fin(BopData *);
BopStatus bop_set_n(long, BopData *);
BopStatus bop_set_vars(int, const char *, BopData *);
BopStatus bop_set_type(BopType, BopData *);
BopStatus bop_get_type(const BopData *, BopType *);
void *bop_get_data(BopData *);
BopStatus bop_write_header(MPI_Comm, const char *, const BopData *);
BopStatus bop_write_values(MPI_Comm, const char *, const BopData *);
BopStatus bop_summary(const BopData *);

struct BopData;
#ifndef UDX_IO_OUTPUT_COMMON_LIMITS
#define UDX_IO_OUTPUT_COMMON_LIMITS
enum { N_MAX = 100 };
#endif

struct IOPointConf {
    int nn[N_MAX];
    char keys[N_MAX][FILENAME_MAX];
    int i;
};

#ifndef UDX_IO_OUTPUT_COMMON_UNSET
#define UDX_IO_OUTPUT_COMMON_UNSET
enum { UNSET = - 1 };
#endif
struct IOPoint {
    int maxn;
    int n;

    int nkey;
    int cum_n;
    int nn[N_MAX];
    int seen[N_MAX];
    char keys[N_MAX][FILENAME_MAX];

    char path[FILENAME_MAX];
    BopData *bop;
};
#ifndef UDX_IO_OUTPUT_COMMON_UTIL
#define UDX_IO_OUTPUT_COMMON_UTIL
static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }
#endif

static void mkdir(const char *p, const char *s) {
    char path[FILENAME_MAX] = {'\0'};
    cpy(path, p);
    cat(path, "/");
    cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}

static void reset(IOPoint *q) {
    int i, nkey;
    nkey = q->nkey;
    for (i = 0; i < nkey; i++)
        q->seen[i] = 0;
    q->n = UNSET;
}

static void wrong_key(IOPoint *q, const char *key) {
    int i, nkey;
    nkey = q->nkey;
    msg_print("unkown key: '%s'", key);
    msg_print("possible values:");
    for (i = 0; i < nkey; i++)
        msg_print("'%s'", q->keys[i]);
    ERR("");
}
#define PATTERN "%s/%s/%05d"

void io_point_conf_ini(     IOPointConf **pq) {
    IOPointConf *q;
    EMALLOC(1, &q);
    q->i = 0;
    *pq = q;
}

static void conf_push(IOPointConf *q, int nv, const char *k0) {
    int i;
    i = q->i;
    cpy(q->keys[i], k0);
    q->nn[i] = nv; q->i = i + 1;
}
void io_point_conf_push(IOPointConf *q, const char *key) {
    int nw;
    UC(nw = string_nword(key));
    msg_print("string, nword: '%s', %d", key, nw);
    conf_push(q, nw, key);
}
void io_point_conf_fin(IOPointConf *q) { EFREE(q); }

static void ini_bop(int maxn, int n, const char *keys, BopData **pq) {
    BopData *q;
    BPC(bop_ini(&q));
    BPC(bop_set_n(maxn, q));
    BPC(bop_set_vars(n, keys, q));
    BPC(bop_set_type(BopDOUBLE, q));
    BPC(bop_alloc(q));

    *pq = q;
}
void io_point_ini(int maxn, const char *path, IOPointConf *c,      IOPoint **pq) {
    int i, nkey, cum_n;
    IOPoint *q;
    char cum_key[N_MAX*(FILENAME_MAX + 1)];
    EMALLOC(1, &q);
    nkey = c->i;
    reset(q);

    cum_n = 0;
    for (i = 0; i < nkey; i++) {
        q->nn[i] = c->nn[i];
        cum_n += q->nn[i];
        cpy(q->keys[i], c->keys[i]);
    }

    for (i = 0; i < nkey; i++) {
        if (i > 0) cat(cum_key, " ");
        cat(cum_key, c->keys[i]);
    }

    ini_bop(maxn, cum_n, cum_key, &q->bop);

    UC(mkdir(DUMP_BASE, path));
    cpy(q->path, path);
    q->nkey = nkey;
    q->maxn = maxn;
    q->cum_n = cum_n;
    *pq = q;
}

void io_point_fin(IOPoint *q) {
    BPC(bop_fin(q->bop));
    EFREE(q);
}

static double *get_data(BopData *b) {
    BopType type;
    BPC(bop_get_type(b, &type));
    if (type != BopDOUBLE) ERR("BopType is not double");
    return (double*)bop_get_data(b);
}

static void push(int n, int nvar, int cum_n, const double *F, int offset, double *T) {
    int i, j, f, t;
    for (f = i = 0; i < n; i++) {
        for (j = 0; j < nvar; j++) {
            t = cum_n*i + offset + j;
            T[t] = F[f++];
        }
    }
}
void io_point_push(IOPoint *q, int n, const double *D, const char *key) {
    int offset, nkey, i;
    double *B;
    if (n > q->maxn)
        ERR("n=%d > q->maxn=%d; key: '%s'",
            n, q->maxn, key);

    if (q->n == UNSET)
        q->n = n;
    else if (q->n != n)
        ERR("q->n=%d != n=%d", q->n, n);

    nkey = q->nkey;
    offset = 0;
    for (i = 0; ; i++) {
        if (i == nkey) {UC(wrong_key(q, key)); ERR(""); }
        if (same_str(key, q->keys[i])) break;
        offset += q->nn[i];
    }
    if (q->seen[i]) ERR("seen key '%s' ", key);
    q->seen[i] = 1;
    UC(B = get_data(q->bop));
    push(n, q->nn[i], q->cum_n, D, offset,      B);
}

void io_point_write(IOPoint *q, MPI_Comm comm, int id) {
    char name[FILENAME_MAX];
    int i, nkey, n;
    nkey = q->nkey;
    n = q->n;
    for (i = 0; i < nkey; i++)
        if (!q->seen[i])
            ERR("key '%s' was not pushed", q->keys[i]);

    if (snprintf(name, FILENAME_MAX, PATTERN, DUMP_BASE, q->path, id) < 0)
        ERR("snprintf failed");

    BPC(bop_set_n(n, q->bop));
    BPC(bop_write_header(comm, name, q->bop));
    BPC(bop_write_values(comm, name, q->bop));
    BPC(bop_summary(q->bop));

    reset(q);
}






