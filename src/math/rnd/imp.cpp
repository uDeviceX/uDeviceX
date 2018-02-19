#include <limits>
#include <stdio.h>
#include <stdint.h>

#include "utils/error.h"
#include "utils/imp.h"

#include "imp.h"
typedef uint32_t integer;
struct RNDunif { integer x, y, z, c; };

void rnd_ini(int x, int y, int z, int c, /**/ RNDunif **r0) {
    RNDunif *r;
    EMALLOC(1, &r);
    r = *r0;

    r->x = x;
    r->y = y;
    r->z = z;
    r->c = c;
}

void rnd_fin(RNDunif *r) {
    UC(efree(r));
}

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
