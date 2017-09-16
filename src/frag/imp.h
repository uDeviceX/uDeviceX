namespace frag {

/* use macros so we don't need nvcc to compile */
/* see /poc/communication                      */

#define i2d(i, c) {                             \
        ((i)     + 2) % 3 - 1,                  \
        ((i) / 3 + 2) % 3 - 1,                  \
        ((i) / 9 + 2) % 3 - 1}

#define d2i(d) (((d[0] + 2) % 3)                \
                + 3 * ((d[1] + 2) % 3)          \
                + 9 * ((d[2] + 2) % 3))         \

}
