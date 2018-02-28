#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

#include "bop_common.h"
#include "bop_serial.h"

#include "../common/macros.h"

struct Args {
    int color;
    char *pp, *cc, *ii;
    char *out;
};

static void usg() {
    fprintf(stderr, "usg: u.filter.color <color> <out> <pp.bop> <cc.bop> (OPT:) <ii.bop>\n");
    exit(1);
}

static int arg_shift(int *argc, char ***argv) {
    (*argv) ++;
    (*argc) --;
    return (*argc) > 0;
}

static void parse(int argc, char **argv, /**/ Args *a) {
    // skip executable
    if (!arg_shift(&argc, &argv)) usg();
    a->color = atoi((*argv));

    if (!arg_shift(&argc, &argv)) usg();
    a->out = *argv;    

    if (!arg_shift(&argc, &argv)) usg();
    a->pp = *argv;

    if (!arg_shift(&argc, &argv)) usg();
    a->cc = *argv;

    if (arg_shift(&argc, &argv))
        a->ii = *argv;
    else
        a->ii = NULL;
}

static void get_n(int tag, long n, const int *pred, long *nmatch) {
    long i, j;
    for (i = j = 0; i < n; ++i)
        j += pred[i] == tag;
    *nmatch = j;
}

static void collect(int tag, long n, const int *pred, int stride, const float *in, float *out) {
    long i, j, c;
    for (i = j = 0; i < n; ++i) {
        if (pred[i] == tag) {
            for (c = 0; c < stride; ++c)
                out[stride * j + c] = in[stride * i + c];
            ++j;
        }
    }
}

static void filter() {
    
}

int main(int argc, char **argv) {
    Args a;
    BopData *outpp, *pp, *cc, *ii = NULL, *outii = NULL;
    
    parse(argc, argv, /**/ &a);

    BPC( bop_ini(&pp) );
    BPC( bop_ini(&cc) );
    if (a.ii)
        BPC( bop_ini(&ii) );

    
    
    BPC( bop_fin(pp) );
    BPC( bop_fin(cc) );
    if (a.ii)
        BPC( bop_fin(ii) );
    
    return 0;
}
