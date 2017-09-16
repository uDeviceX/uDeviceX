#include <stdio.h>

enum {bulk = 26};

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
#define anti(i) d2i(-i2d((i), 0),               \
                    -i2d((i), 1),               \
                    -i2d((i), 2))

void print_i2d() {
    int i;
    for (i = 0; i < bulk + 1; ++i)
        printf("%2d -- (%2d %2d %2d)\n", i,
               i2d(i, 0), i2d(i, 1), i2d(i, 2));
}

void print_d2i() {
    int x, y, z;
    for (z = 0; z < 3; ++z)
        for (y = 0; y < 3; ++y)
            for (x = 0; x < 3; ++x)
                printf("(%2d %2d %2d) -- %2d\n",
                       x-1, y-1, z-1, d2i(x-1, y-1, z-1));
}

void printanti() {
   int i;
    for (i = 0; i < bulk + 1; ++i)
        printf("%2d -- %2d\n", i, anti(i));
}

int main() {
    printf("\ni2d\n");
    print_i2d();
    printf("\nd2i\n");
    print_d2i();
    printf("\nanti\n");
    printanti();
}
