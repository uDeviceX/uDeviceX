#include <stdio.h>
#include "frag.h"

#define XS 10
#define YS 20
#define ZS 30

int main(int c, char **v) {
    enum {X, Y, Z};
    int id;
    printf("%d\n", id = frag_to_id(0, 0, -1));
    printf("%d %d %d\n", frag_to_dir[id][X], frag_to_dir[id][Y], frag_to_dir[id][Z]);
    printf("%d %d %d\n", frag_fro_dir[id][X], frag_fro_dir[id][Y], frag_fro_dir[id][Z]);
    printf("%d\n", frag_ncell(id));
}
