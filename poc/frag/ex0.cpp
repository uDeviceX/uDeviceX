#include <stdio.h>
#include <stdlib.h>
#include "frag.h"

#define XS 10
#define YS 20
#define ZS 30

int main(int, char **v) {
    enum {X, Y, Z};
    int x, y, z, id, i;
    i = 1;
    x = atoi(v[i++]); y = atoi(v[i++]); z = atoi(v[i++]);
    
    printf("id: %d\n", id = frag_to_id(x, y, z));
    printf("to: % d % d % d\n",  frag_to_dir[id][X],  frag_to_dir[id][Y],  frag_to_dir[id][Z]);
    printf("fro: % d % d % d\n", frag_fro_dir[id][X], frag_fro_dir[id][Y], frag_fro_dir[id][Z]);
    printf("ncell: %d\n", frag_ncell(id));
}
