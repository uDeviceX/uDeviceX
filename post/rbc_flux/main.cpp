#include <stdio.h>
#include <stdlib.h>

#include "type.h"
#include "com.h"

int main(int argc, char **argv ) {

    Com *cc;
    int n;

    FILE *f;

    f = fopen(argv[1], "r");
    read(f, &n, &cc);
    
    sort_by_id(n, cc);

    print_cc(n, cc, stderr);

    fclose(f);
    free(cc);
    
    return 0;
}
