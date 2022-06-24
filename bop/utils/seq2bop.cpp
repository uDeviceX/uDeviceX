#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "bop_common.h"
#include "bop_serial.h"

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr, "usage: ./seq2bop <N> <out.bop>\n");
        exit(1);
    }
    
    int *data, i;
    BopData *d;

    const int N = atoi(argv[1]);

    bop_ini(&d);

    bop_set_n(N, d);
    bop_set_type(BopIASCII, d);
    bop_set_vars(1, "seq", d);

    bop_alloc(d);
    data = (int *) bop_get_data(d);
    
    for (i = 0; i < N; ++i) data[i] = i;

    bop_write_header(argv[2], d);
    bop_write_values(argv[2], d);
    
    bop_fin(d);
    return 0;
}
