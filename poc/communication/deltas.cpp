#include <stdio.h>

void del_end_bulk(int i, int d[3]) {
    d[0] = (i     + 2) % 3 - 1;
    d[1] = (i / 3 + 2) % 3 - 1;
    d[2] = (i / 9 + 2) % 3 - 1;
}

void del_strt_bulk(int i, int d[3]) {
    d[0] = (i     + 1) % 3 - 1;
    d[1] = (i / 3 + 1) % 3 - 1;
    d[2] = (i / 9 + 1) % 3 - 1;
}

bool valid(void (*i2d)(int i, int d[3]) ) {
    int checked[3][3][3] = {0};
    int i, d[3];

    for (i = 0; i < 27; ++i) {
        i2d(i, d);
        checked [d[0] + 1] [d[1] + 1] [d[2] + 1] = 1;
    }

    int *checkp = &checked[0][0][0];
    for (int i = 0; i < 27; ++i, ++checkp)
        if (*checkp != 1)
            return false;
    return true;
}

#define check(F) do {                           \
        if (valid(&F))                          \
            printf(#F " is valid\n");           \
        else                                    \
            printf(#F " is NOT valid\n");       \
    } while (0)


void full_print_(void (*i2d)(int i, int d[3]) ) {
    int i, d[3];

    for (i = 0; i < 27; ++i) {
        i2d(i, d);
        printf("%2d:  %2d %2d %2d\n", i, d[0], d[1], d[2]);
    }
}

#define full_print(F) do {                      \
        printf("\n\n" #F "\n\n");               \
        printf("i :  dx dy dz\n");              \
        full_print_(&F);                        \
    } while(0)

int main() {

    check(del_end_bulk);
    check(del_strt_bulk);

    full_print(del_end_bulk);
    full_print(del_strt_bulk);
    
    return 0;
}
