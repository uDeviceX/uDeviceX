// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/algo/key_list/imp.cpp */
#include <stdio.h>
#include <string.h>

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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
/* begin src/utils/string/imp.h */
int string_nword(const char *s);
/* end src/utils/string/imp.h */

/* begin src/algo/key_list/imp.h */
struct KeyList;

void key_list_ini(KeyList**);
void key_list_copy(KeyList*, /**/ KeyList**);
void key_list_fin(KeyList*);

void key_list_push(KeyList*, const char*);
int key_list_has(KeyList*, const char*);
int key_list_offset(KeyList*, const char*);
int key_list_width(KeyList*, const char*);
int key_list_size(KeyList*);
const char* key_list_name(KeyList*, int i);

void key_list_mark(KeyList*, const char*);
int  key_list_marked(KeyList*);
void key_list_unmark(KeyList*);

void key_list_log(KeyList*);
/* end src/algo/key_list/imp.h */
/* begin src/algo/key_list/imp/type.h */
enum {MAX_NK = 100 };
enum {STAMP_GOOD = 42 };

struct KeyList {
    int nk; /* number of keys */
    int  ww[MAX_NK]; /* width */
    int  mark[MAX_NK];
    char keys[MAX_NK][FILENAME_MAX];
    int stamp;
};
/* end src/algo/key_list/imp/type.h */
/* begin src/algo/key_list/imp/util.h */
static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
/* end src/algo/key_list/imp/util.h */
/* begin src/algo/key_list/imp/main.h */
void key_list_ini(KeyList **pq) {
    KeyList *q;
    EMALLOC(1, &q);
    q->nk = 0;
    q->stamp = STAMP_GOOD;
    *pq = q;
}

void key_list_copy(KeyList *a, /**/ KeyList **pq) {
    int i, nk;
    KeyList *q;
    EMALLOC(1, &q);
    nk = a->nk;
    for (i = 0; i < nk; i++) {
        q->ww[i] = a->ww[i];
        q->mark[i] = a->mark[i];
        cpy(q->keys[i], a->keys[i]);
    }
    q->stamp = STAMP_GOOD;
    q->nk = nk;
    *pq = q;
}

void key_list_fin(KeyList *q) {
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    EFREE(q);
}

void key_list_push(KeyList *q, const char *k) {
    int nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    if (nk == MAX_NK)
        ERR("nk=%d == MAX_NK=%d", nk, MAX_NK);
    cpy(q->keys[nk], k);
    q->ww[nk] = string_nword(k);
    q->mark[nk] = 0;
    q->nk = nk + 1;
}

int key_list_has(KeyList *q, const char *k) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (same_str(q->keys[i], k)) return 1;
    return 0;
}

int key_list_offset(KeyList *q, const char *k) {
    int i, nk, offset;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = offset = 0; i < nk; i++) {
        if (same_str(q->keys[i], k)) return offset;
        offset += q->ww[i];
    }
    ERR("key `%s` is not found", k);
    return -1;
}

int key_list_width(KeyList *q, const char *k) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++) {
        if (same_str(q->keys[i], k)) return q->ww[i];
    }
    ERR("key `%s` is not found", k);
    return -1;
}

int key_list_size(KeyList *q) {
    int i, nk, size;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = size = 0; i < nk; i++)
        size += q->ww[i];
    return size;
}

const char* key_list_name(KeyList *q, int i) {
    int n;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    n = key_list_size(q);
    if (i < 0) ERR("i=%d < 0", i);
    if (i >= n) ERR("i=%d >= n=%d", i, n);
    return q->keys[i];
}

void key_list_mark(KeyList *q, const char *k) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (same_str(q->keys[i], k)) {
            q->mark[i] = 1;
            return;
        }
    ERR("key `%s` is not found", k);
}

int  key_list_marked(KeyList *q) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (q->mark[i] == 0) return 0;
    return 1;
}

void  key_list_unmark(KeyList *q) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        q->mark[i] = 0;
}

void key_list_log(KeyList *q) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    msg_print("<key_list");
    for (i = 0; i < nk; i++)
        msg_print("'%s' %d", q->keys[i], q->mark[i]);
    msg_print(">");
}
/* end src/algo/key_list/imp/main.h */
/* end src/algo/key_list/imp.cpp */

#line 1 "src/algo/minmax/imp.cu"
