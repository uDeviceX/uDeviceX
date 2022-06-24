#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bop_common.h"
#include "bop_serial.h"
#include "check.h"

template <typename real>
void float_print(const real *data, const long n, const int nvars) {
    for (long i = 0; i < n; ++i) {
        for (int j = 0; j < nvars; ++j)
        printf("%.6e ", data[nvars*i + j]);
        printf("\n");
    }
}

void int_print(const int *data, const long n, const int nvars) {
    for (long i = 0; i < n; ++i) {
        for (int j = 0; j < nvars; ++j)
        printf("%d ", data[nvars*i + j]);
        printf("\n");
    }
}

int main(int argc, char **argv) {
    char dfname[256];
    BopData *d;
    BopType type;
    bool summary = false;
    int istrt, i, nvars;
    long n;
    if (argc < 2) {
        fprintf(stderr, "usage: %s <OPT> <in1.bop> <in2.bop> ...\n", argv[0]);
        fprintf(stderr, "\tOPT: -s for summary");
        
        exit(1);
    }

    istrt = 1;
    if (strcmp(argv[1], "-s") == 0) {
        ++istrt;
        summary = true;
    }

    for (i = istrt; i < argc; ++i) {

        BPC(bop_ini(&d));
        BPC(bop_read_header(argv[i], /**/ d, dfname));
        BPC(bop_alloc(d));
        BPC(bop_read_values(dfname, /**/ d));
        if (summary)
            BPC(bop_summary(d));

        BPC(bop_get_type(d, &type));
        BPC(bop_get_n(d, &n));
        BPC(bop_get_nvars(d, &nvars));
        
        switch (type) {
        case BopFLOAT:
        case BopFASCII:
            float_print((const float *) bop_get_data(d), n, nvars);
            break;
        case BopDOUBLE:
            float_print((const double *) bop_get_data(d), n, nvars);
            break;
        case BopINT:
        case BopIASCII:
            int_print((const int *) bop_get_data(d), n, nvars);
            break;
        };
        BPC(bop_fin(d));
    }    
    return 0;
}

/*

# nTEST: bop2txt.t0
# make 
# ./bop2txt data/test.bop > test.out.txt

*/
