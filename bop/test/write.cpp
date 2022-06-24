#include <stdio.h>
#include <stdlib.h>

#include "bop_common.h"
#include "bop_serial.h"
#include "check.h"

int main(int argc, char **argv) {

    BopData *d;
    int i, n;
    float *data;

    n = 10;
    
    BPC(bop_ini(&d));
    BPC(bop_set_n(n, d));
    BPC(bop_set_type(BopFLOAT, d));
    BPC(bop_set_vars(2, "x y", d));
    BPC(bop_alloc(d));

    data = (float*) bop_get_data(d);

    for (i = 0; i < n; ++i) {
        data[2*i + 0] = i * 0.5;
        data[2*i + 1] = n - i * 0.5;
    }

    BPC(bop_write_header("test", d));
    BPC(bop_write_values("test", d));
    
    BPC(bop_fin(d));
    
    return 0;
}


/*

  # TEST: write.t0
  # ./write
  # bop2txt test.bop > test.out.txt

*/
