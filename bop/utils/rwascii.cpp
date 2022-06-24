#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "bop_common.h"
#include "bop_serial.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: ./%s <in.bop> <out.bop>\n", argv[0]);
        exit(1);
    }

    BopData *d;
    char dfname[256];

    bop_ini(&d);
    bop_read_header(argv[1], /**/ d, dfname);
    bop_alloc(d);
    bop_read_values(dfname, /**/ d);

    bop_summary(d);

    bop_write_header(argv[2], d);
    bop_write_values(argv[2], d);
    
    bop_fin(d);
    
    return 0;
}

/*

  # nTEST: ascii2ascii.t0
  # make -j 
  # ./rwascii data/ascii.bop test
  # mv test.values test.out.txt

  # nTEST: ascii2ascii.t1
  # make -j 
  # ./rwascii data/iascii.bop test
  # mv test.values test.out.txt

*/
