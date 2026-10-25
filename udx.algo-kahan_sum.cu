// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/algo/kahan_sum/imp.cpp */
#include <stdio.h>

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

/* begin src/algo/kahan_sum/imp.h */
struct KahanSum;

// tag::interface[]
void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input); // <1>
double kahan_sum_get(const KahanSum*); // <2>
double kahan_sum_compensation(const KahanSum*); // <3>
// end::interface[]
/* end src/algo/kahan_sum/imp.h */
/* begin src/algo/kahan_sum/imp/type.h */
/*

var sum = 0.0
    var c = 0.0
    for i = 1 to input.length do
        var y = input[i] - c
        var t = sum + y
        c = (t - sum) - y
        sum = t
    next i
    return sum

 */

struct KahanSum {
    double c, y, sum;
};
/* end src/algo/kahan_sum/imp/type.h */
/* begin src/algo/kahan_sum/imp/main.h */
void kahan_sum_ini(/**/ KahanSum **pq) {
    KahanSum *q;
    EMALLOC(1, &q);
    q->c = q->sum = 0;
    *pq = q;
}

void kahan_sum_fin(KahanSum *q) { EFREE(q); }
void kahan_sum_add(KahanSum *q, double input) {
    double t;
    double y, c, sum;
    y = q->y; c = q->c; sum = q->sum;

    y = input - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;

    q->y = y; q->c = c; q->sum = sum;
}

double kahan_sum_get(const KahanSum *q) { return q->sum; }
double kahan_sum_compensation(const KahanSum *q) { return q->c; }
/* end src/algo/kahan_sum/imp/main.h */
/* end src/algo/kahan_sum/imp.cpp */

#line 1 "src/algo/key_list/imp.cpp"
