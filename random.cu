#include "macros.cuh"
#include <limits>
#include <stdint.h>

void emalloc(size_t, void**);
void efree(void*);
void error_stack_push(const char *file, int line);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();


typedef uint32_t integer;
struct RNDunif { integer x, y, z, c; };

void rnd_ini(int x, int y, int z, int c,      RNDunif **pq) {
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
    r->y ^= ( r->y << 5 );
    t = a * r->z + r->c;
    r->c = ( t >> 32 );
    return r->x + r->y + ( r->z = t );
}

float rnd_get(RNDunif *r) {
    return get_int(r) / float( std::numeric_limits<integer>::max() );
}
