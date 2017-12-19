#include <limits>
#include <stdint.h>
#include "rnd/imp.h"

// namespace rnd {
// KISS::KISS(integer x_, integer y_, integer z_, integer c_) {
//     x = x_; y = y_; z = z_; c = c_;
// }

// float KISS::get_float() {
//     return get_int() / float( std::numeric_limits<integer>::max() );
// }

// KISS::integer KISS::get_int() {
//     uint64_t t, a = 698769069ULL;
//     x = 69069 * x + 12345;
//     y ^= ( y << 13 );
//     y ^= ( y >> 17 );
//     y ^= ( y << 5 ); /* y must never be set to zero! */
//     t = a * z + c;
//     c = ( t >> 32 ); /* Also avoid setting z=c=0! */
//     return x + y + ( z = t );
// }
// }
