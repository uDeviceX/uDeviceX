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
#define d2i(d) (((d[0] + 2) % 3)                \
                + 3 * ((d[1] + 2) % 3)          \
                + 9 * ((d[2] + 2) % 3))         \


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


}
