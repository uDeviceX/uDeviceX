namespace frag {

enum {frag_bulk = 26};

/* use macros so we don't need nvcc to compile */
/* see /poc/communication                      */

/* fragment id to direction                    */
#define i2d(i, c) (                             \
                   c == 0 ?                     \
                   (((i)     + 2) % 3 - 1) :    \
                   c == 1 ?                     \
                   (((i) / 3 + 2) % 3 - 1) :    \
                   (((i) / 9 + 2) % 3 - 1))

/* direction to fragment id                    */
#define d2i(x, y, z) ((((x) + 2) % 3)           \
                      + 3 * (((y) + 2) % 3)     \
                      + 9 * (((z) + 2) % 3))    \


/* number of cells in direction x, y, z        */
#define ncell0(x, y, z)                         \
    ((((x) == 0 ? (XS) : 1)) *                  \
     (((y) == 0 ? (YS) : 1)) *                  \
     (((z) == 0 ? (ZS) : 1)))

/* number of cells in fragment i               */
#define ncell(i)                                \
    (ncell0(i2d(i, 0),                          \
            i2d(i, 1),                          \
            i2d(i, 2)))

/* anti direction to fragment id                */
#define ad2i(x, y, z) d2i((-x), (-y), (-z))

/* anti fragment                                */
#define anti(i) f2i(-i2d((i), 0),               \
                    -i2d((i), 1),               \
                    -i2d((i), 2))

}
