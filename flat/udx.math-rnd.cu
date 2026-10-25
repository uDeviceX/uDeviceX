// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/math/rnd/imp.cpp */
#include <limits>
#include <stdio.h>
#include <stdint.h>

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

/* begin src/math/rnd/imp.h */
// tag::interface[]
struct RNDunif;                                             // <1>
void rnd_ini(int x, int y, int z, int c, /**/ RNDunif **r); // <2>
void rnd_fin(RNDunif *r);                                   // <3>
float rnd_get(RNDunif *r);                                  // <4>
// end::interface[]
/* end src/math/rnd/imp.h */
typedef uint32_t integer;
struct RNDunif { integer x, y, z, c; };

void rnd_ini(int x, int y, int z, int c, /**/ RNDunif **pq) {
    RNDunif *q;
    EMALLOC(1, &q);
    *pq = q;
    q->x = x; q->y = y; q->z = z; q->c = c;
}

void rnd_fin(RNDunif *r) { EFREE(r); }

static integer get_int(RNDunif *r) {
    uint64_t t, a = 698769069ULL;
    r->x = 69069 * r->x + 12345;
    r->y ^= ( r->y << 13 );
    r->y ^= ( r->y >> 17 );
    r->y ^= ( r->y << 5 ); /* y must never be set to zero! */
    t = a * r->z + r->c;
    r->c = ( t >> 32 ); /* Also avoid setting z=c=0! */
    return r->x + r->y + ( r->z = t );
}

float rnd_get(RNDunif *r) {
    return get_int(r) / float( std::numeric_limits<integer>::max() );
}
/* end src/math/rnd/imp.cpp */

#line 1 "src/math/tform/imp.cpp"
