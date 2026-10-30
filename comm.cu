#include "memory.cuh"
#include "macros.cuh"
#include "device.cuh"
#include <mpi.h>
#include <stdio.h>
#include <string.h>

typedef unsigned char data_t;
enum { NFRAGS = 26, NBAGS = 27, frag_bulk = 26 };
enum AllocMod { HST_ONLY, DEV_ONLY, PINNED, PINNED_HST, PINNED_DEV, NONE };
struct dBags { data_t *data[NBAGS]; };
struct hBags { data_t *data[NBAGS]; int *counts; int capacity[NBAGS]; size_t bsize; };
struct Comm { MPI_Request sreq[NBAGS]; MPI_Request rreq[NBAGS]; MPI_Comm cart; int ranks[NFRAGS]; int tags[NFRAGS]; };
struct CommBuffer { data_t *buf[NBAGS]; size_t sz[NBAGS], cap[NBAGS]; };

namespace m {
int Cart_get(MPI_Comm, int, int[], int[], int[]);
int Cart_rank(MPI_Comm, const int[], int*);
int Comm_dup(MPI_Comm, MPI_Comm*);
int Comm_free(MPI_Comm*);
int Irecv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*);
int Isend(const void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*);
int Waitall(int, MPI_Request[], MPI_Status[]);
int Get_count(const MPI_Status*, MPI_Datatype, int*);
int Error_string(int, char*, int*);
bool is_success(int);
bool is_pending(int);
bool is_err_in_status(int);
int status2errcode(MPI_Status*);
}
namespace frag_hst { void i2d3(int, int[3]); int anti(int); }
namespace cc { void check(int, const char*, int); }
#define MC(ans) do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)
#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)
namespace mpicheck { void check(int, const char*, int); }
void error_stack_push(const char*, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char*, int, const char*, ...);

static void alloc_counts(int n,      int **hc) {
    size_t sz = n * sizeof(int);
    EMALLOC(n, hc);
    memset(*hc, 0, sz);
}


static void alloc_pair(int i, AllocMod mod,      hBags *hb, dBags *db) {
    size_t n = hb->capacity[i] * hb->bsize;

    switch (mod) {
    case HST_ONLY:
        if (n) UC(emalloc(n, (void**) &hb->data[i]));
        else   hb->data[i] = NULL;
        break;
    case DEV_ONLY:
        if (n) CC(d::Malloc((void**) &db->data[i], n));
        else db->data[i] = NULL;
        break;
    case PINNED:
        if (n) {
            CC(d::alloc_pinned((void**) &hb->data[i], n));
            CC(d::HostGetDevicePointer((void**) &db->data[i], hb->data[i], 0));
        } else {
            hb->data[i] = db->data[i] = NULL;
        }
        break;
    case PINNED_HST:
        if (n) CC(d::alloc_pinned((void**) &hb->data[i], n));
        else hb->data[i] = NULL;
        break;
    case PINNED_DEV:
        if (n) {
            CC(d::alloc_pinned((void**) &hb->data[i], n));
            CC(d::Malloc((void **) &db->data[i], n));
        } else {
            hb->data[i] = db->data[i] = NULL;
        }
        break;
    case NONE:
    default:
        break;
    }
}

void comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS],      hBags *hb, dBags *db) {
    hb->bsize = bsize;
    memcpy(hb->capacity, capacity, NBAGS * sizeof(int));


    for (int i = 0; i < NFRAGS; ++i)
        UC(alloc_pair(i, fmod,      hb, db));


    UC(alloc_pair(frag_bulk, bmod,      hb, db));

    UC(alloc_counts(NBAGS,      &hb->counts));
}

static void comm_buffer_ini_one_frag(int fid, int nbags, const hBags *hbb, CommBuffer *b) {
    size_t bs, cap;
    int i;
    cap = nbags * sizeof(int);
    for (i = 0; i < nbags; ++i) {
        bs = hbb[i].bsize;
        cap += hbb[i].capacity[fid] * bs;
    }
    EMALLOC(cap, &b->buf[fid]);
    b->cap[fid] = cap;
    b->sz[fid] = 0;
}

void comm_buffer_ini(int nbags, const hBags *hbb, CommBuffer **cb) {
    EMALLOC(1, cb);
    for (int i = 0; i < NFRAGS; ++i)
        comm_buffer_ini_one_frag(i, nbags, hbb, *cb);
}

void comm_ini(MPI_Comm cart,      Comm **com_p) {
    int i, c, crd_rnk[3], d[3];
    int coords[3], periods[3], dims[3];
    Comm *com;

    EMALLOC(1, com_p);
    com = *com_p;

    MC(m::Cart_get(cart, 3, dims, periods, coords));

    for (i = 0; i < NFRAGS; ++i) {
        frag_hst::i2d3(i, d);
        for (c = 0; c < 3; ++c)
            crd_rnk[c] = coords[c] + d[c];
        MC(m::Cart_rank(cart, crd_rnk, com->ranks + i));
        com->tags[i] = frag_hst::anti(i);
    }
    MC(m::Comm_dup(cart, &com->cart));
}

static void free_counts(int **hc) {
    EFREE(*hc);
    *hc = NULL;
}

static void free_pair(int i, AllocMod mod,      hBags *hb, dBags *db) {
    switch (mod) {
    case HST_ONLY:
        if (hb->data[i])
            EFREE(hb->data[i]);
        break;
    case DEV_ONLY:
        if (db->data[i])
            CC(d::Free(db->data[i]));
        break;
    case PINNED_HST:
    case PINNED:
        if (hb->data[i])
            CC(d::FreeHost(hb->data[i]));
        break;
    case PINNED_DEV:
        if (hb->data[i])
            CC(d::FreeHost(hb->data[i]));
        if (db->data[i])
            CC(d::Free(db->data[i]));
        break;
    case NONE:
    default:
        break;
    }
}

void comm_bags_fin(AllocMod fmod, AllocMod bmod,      hBags *hb, dBags *db) {

    for (int i = 0; i < NFRAGS; ++i)
        free_pair(i, fmod,      hb, db);


    free_pair(frag_bulk, bmod,      hb, db);
    free_counts(     &hb->counts);
}

void comm_buffer_fin(CommBuffer *cb) {
    int i;
    for (i = 0; i < NFRAGS; ++i)
        EFREE(cb->buf[i]);
    EFREE(cb);
}

void comm_fin(     Comm *c) {
    MC(m::Comm_free(&c->cart));
    EFREE(c);
}
static void frag_set_buffer(int fid, int nbags, const hBags *hbb, CommBuffer *cb) {
    size_t tot_sz, sz, bs;
    int i, c, *counts;
    data_t *d;
    const hBags *hb;

    d = cb->buf[fid];
    counts = (int*) d;
    tot_sz = sz = nbags * sizeof(int);
    d += sz;

    for (i = 0; i < nbags; ++i) {
        hb = &hbb[i];
        bs = hb->bsize;
        c  = hb->counts[fid];
        sz = c * bs;
        memcpy(d, hb->data[fid], sz);
        counts[i] = c;
        d += sz;
        tot_sz += sz;
    }
    cb->sz[fid] = tot_sz;

    if (tot_sz > cb->cap[fid]) ERR("fragment %d: exceed buffer size %ld/%ld\n", fid, tot_sz, cb->cap[fid]);
}

void comm_buffer_set(int nbags, const hBags *hbb, CommBuffer *cb) {
    for (int fid = 0; fid < NFRAGS; ++fid)
        frag_set_buffer(fid, nbags, hbb, cb);
}

static void frag_get_buffer(int fid, const CommBuffer *cb, int nbags, hBags *hbb) {
    size_t sz;
    int i, c, *counts;
    data_t *d;
    hBags *hb;

    d = cb->buf[fid];
    counts = (int*) d;
    d += nbags * sizeof(int);

    for (i = 0; i < nbags; ++i) {
        hb = &hbb[i];
        c = counts[i];
        hb->counts[fid] = c;
        sz = c * hb->bsize;
        memcpy(hb->data[fid], d, sz);
        d += sz;
    }
}

void comm_buffer_get(const CommBuffer *cb, int nbags, hBags *hbb) {
    for (int fid = 0; fid < NFRAGS; ++fid)
        frag_get_buffer(fid, cb, nbags, hbb);
}
void comm_post_recv(hBags *b, Comm *com) {
    int i, c, tag;
    for (i = 0; i < NFRAGS; ++i) {
        c = b->capacity[i] * b->bsize;
        tag = com->tags[i];
        MC(m::Irecv(b->data[i], c, MPI_BYTE, com->ranks[i], tag, com->cart, com->rreq + i));
    }
}

static void fail_over(int i, long c, long cap) {
    enum {X, Y, Z};
    int d[3];
    frag_hst::i2d3(i, d);
    ERR("over capacity, fragment %d = [%d %d %d]: %ld/%ld",
        i, d[X], d[Y], d[Z], c, cap);
}

void comm_post_send(const hBags *b, Comm *com) {
    int i, n, c, cap, tag;
    for (i = 0; i < NFRAGS; ++i) {
        c = b->counts[i];
        cap = b->capacity[i];
        n = c * b->bsize;
        tag = i;
        if (c > cap) UC(fail_over(i, c, cap));
        MC(m::Isend(b->data[i], n, MPI_BYTE, com->ranks[i], tag, com->cart, com->sreq + i));
    }
}

static void get_counts_bytes(const MPI_Status ss[NFRAGS],      int counts[NFRAGS]) {
    for (int i = 0; i < NFRAGS; ++i)
        MC(m::Get_count(ss + i, MPI_BYTE, counts + i));
}

static void get_counts(const MPI_Status ss[NFRAGS],      hBags *b) {
    int cap, i, c, cc[NFRAGS];
    get_counts_bytes(ss,      cc);

    for (i = 0; i < NFRAGS; ++i) {
        c = cc[i] / b->bsize;
        b->counts[i] = c;
        cap = b->capacity[i];
        if (c > cap) UC(fail_over(i, c, cap));
    }
}

static void get_counts(const MPI_Status ss[NFRAGS],      CommBuffer *b) {
    int i, cc[NFRAGS];
    size_t cap, c;
    get_counts_bytes(ss,      cc);

    for (i = 0; i < NFRAGS; ++i) {
        c = cc[i];
        cap = b->cap[i];
        b->sz[i] = c;
        if (c > cap) UC(fail_over(i, c, cap));
    }
}

static void fail_wait_normal(int code) {
    int sz;
    char msg[BUFSIZ];
    m::Error_string(code, msg, &sz);
    ERR(msg);
}

static void fail_wait_status(int n, MPI_Status *ss) {

    enum {X, Y, Z};
    int i, sz, code, d[3];
    char msg[BUFSIZ];
    for (i = 0; i < n; i++) {
        code = m::status2errcode(&ss[i]);
        if (m::is_success(code) || m::is_pending(code)) continue;
        frag_hst::i2d3(i, d);
        m::Error_string(code, msg, &sz);
        ERR("mpi error in fragment %d = [%d %d %d], %s", i,
            d[X], d[Y], d[Z], msg);
    }
    ERR("assert");
}

static void fail_wait(int code, int n, MPI_Status *ss) {
    if (m::is_err_in_status(code)) UC(fail_wait_status(n, ss));
    else  UC(fail_wait_normal(code));
}

void comm_wait_recv(Comm *com,      hBags *b) {
    int errorcode;
    MPI_Status ss[NFRAGS];
    errorcode = m::Waitall(NFRAGS, com->rreq,      ss);
    if (!m::is_success(errorcode)) fail_wait(errorcode, NFRAGS, ss);
    get_counts(ss,      b);
}

void comm_post_recv(CommBuffer *cb, Comm *com) {
    int i, tag;
    for (i = 0; i < NFRAGS; ++i) {
        tag = com->tags[i];
        MC(m::Irecv(cb->buf[i], cb->cap[i], MPI_BYTE, com->ranks[i], tag, com->cart, com->rreq + i));
    }
}

void comm_post_send(const CommBuffer *cb, Comm *com) {
    int i, tag;
    long c, cap;
    for (i = 0; i < NFRAGS; ++i) {
        c = cb->sz[i];
        cap = cb->cap[i];
        tag = i;
        if (c > cap) UC(fail_over(i, c, cap));
        MC(m::Isend(cb->buf[i], c, MPI_BYTE, com->ranks[i], tag, com->cart, com->sreq + i));
    }
}


void comm_wait_recv(Comm *com,      CommBuffer *cb) {
    int errorcode;
    MPI_Status ss[NFRAGS];
    errorcode = m::Waitall(NFRAGS, com->rreq,      ss);
    if (!m::is_success(errorcode)) fail_wait(errorcode, NFRAGS, ss);
    get_counts(ss,      cb);
}

void comm_wait_send(Comm *com) {
    MPI_Status ss[NFRAGS];
    MC(m::Waitall(NFRAGS, com->sreq, ss));
}



int comm_get_number_capacity(int i, const hBags *b) {
    return b->capacity[i];
}

size_t comm_get_byte_capacity(int i, const hBags *b) {
    return b->bsize * comm_get_number_capacity(i, b);
}
