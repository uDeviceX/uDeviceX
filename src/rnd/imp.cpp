#include <limits>
#include <stdint.h>
#include "rnd/imp.h"

namespace rnd {
float KISS::get_float() {
    return get_int() / float( std::numeric_limits<integer>::max() );
}

KISS::integer KISS::get_int() {
    uint64_t t, a = 698769069ULL;
    x = 69069 * x + 12345;
    y ^= ( y << 13 );
    y ^= ( y >> 17 );
    y ^= ( y << 5 ); /* y must never be set to zero! */
    t = a * z + c;
    c = ( t >> 32 ); /* Also avoid setting z=c=0! */
    return x + y + ( z = t );
}
}
