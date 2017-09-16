enum {frag_bulk = 26};

/* use macros so we don't need nvcc to compile */
/* see /poc/communication                      */

/* fragment id to direction                    */
#define frag_i2d(i, c) (c == 0 ?                        \
                        (((i)     + 2) % 3 - 1) :       \
                        c == 1 ?                        \
                        (((i) / 3 + 2) % 3 - 1) :       \
                        (((i) / 9 + 2) % 3 - 1))

#define frag_i2d3(i) { frag_i2d((i), 0),        \
            frag_i2d((i), 1),                   \
            frag_i2d((i), 2)}

/* direction to fragment id                    */
#define frag_d2i(x, y, z) ((((x) + 2) % 3)              \
                           + 3 * (((y) + 2) % 3)        \
                           + 9 * (((z) + 2) % 3))

/* number of cells in direction x, y, z        */
#define frag_ncell0(x, y, z)                    \
    ((((x) == 0 ? (XS) : 1)) *                  \
     (((y) == 0 ? (YS) : 1)) *                  \
     (((z) == 0 ? (ZS) : 1)))

/* number of cells in fragment i               */
#define frag_ncell(i)                           \
    (frag_ncell0(frag_i2d(i, 0),                \
                 frag_i2d(i, 1),                \
                 frag_i2d(i, 2)))

/* anti direction to fragment id                */
#define frag_ad2i(x, y, z) frag_d2i((-x), (-y), (-z))

/* anti fragment                                */
#define frag_anti(i) frag_d2i(-frag_i2d((i), 0),        \
                              -frag_i2d((i), 1),        \
                              -frag_i2d((i), 2))


/* fill capacities given a maximum density        */
void frag_estimates(int nfrags, float maxd, /**/ int *cap);
