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
        j += (pred[i] == tag);
    *nmatch = j;
}

template <typename T>
static void collect(int tag, long n, const int *pred, int stride, const T *in, T *out) {
    long i, j, c;
    for (i = j = 0; i < n; ++i) {
        if (pred[i] == tag) {
            for (c = 0; c < stride; ++c)
                out[stride * j + c] = in[stride * i + c];
            ++j;
        }
    }
}

static void filter(int tag, const BopData *cc, const BopData *pp, const BopData *ii,
                   BopData *outpp, BopData *outii) {
    long n, nout;
    const int *pred;

    BPC( bop_get_n(cc, &n) );
    pred = (const int*) bop_get_data(cc);
    get_n(tag, n, pred, /**/ &nout);

    BPC( bop_set_n(nout, outpp) );
    BPC( bop_set_vars(6, "x y z vx vy vz", outpp) );
    BPC( bop_set_type(BopFLOAT, outpp) );
    BPC( bop_alloc(outpp) );

    const float *pp_in  = (const float*) bop_get_data(pp);
    float       *pp_out =       (float*) bop_get_data(outpp);

    collect(tag, n, pred, 6, pp_in, /**/ pp_out);

    if (ii) {
        BPC( bop_set_n(nout, outii) );
        BPC( bop_set_vars(1, "ids", outii) );
        BPC( bop_set_type(BopINT, outii) );
        BPC( bop_alloc(outii) );

        const int *ii_in  = (const int*) bop_get_data(ii);
        int       *ii_out =       (int*) bop_get_data(outii);

        collect(tag, n, pred, 1, ii_in, /**/ ii_out);        
    }
}

static void read(Args a, BopData *cc, BopData *pp, BopData *ii) {
    char fdname[FILENAME_MAX];

    BPC( bop_read_header(a.cc, cc, fdname) );
    BPC( bop_alloc(cc) );
    BPC( bop_read_values(fdname, cc) );

    BPC( bop_read_header(a.pp, pp, fdname) );
    BPC( bop_alloc(pp) );
    BPC( bop_read_values(fdname, pp) );

    if (a.ii) {
        BPC( bop_read_header(a.ii, ii, fdname) );
        BPC( bop_alloc(ii) );
        BPC( bop_read_values(fdname, ii) );
    }
}

static void getfname(const char *base, const char *specific, char *name) {
    strcpy(name, base);
    strcat(name, specific);
}

static void write(Args a, const BopData *pp, const BopData *ii) {
    char fname[FILENAME_MAX];

    getfname(a.out, ".pp", fname);
    BPC( bop_write_header(fname, pp) );
    BPC( bop_write_values(fname, pp) );

    if (a.ii) {
        getfname(a.out, ".ii", fname);
        BPC( bop_write_header(fname, ii) );
        BPC( bop_write_values(fname, ii) );
    }
}

int main(int argc, char **argv) {
    Args a;
    BopData *outpp, *pp, *cc, *ii = NULL, *outii = NULL;
    
    parse(argc, argv, /**/ &a);

    BPC( bop_ini(&cc) );
    BPC( bop_ini(&pp) );
    BPC( bop_ini(&outpp) );
    if (a.ii) {
        BPC( bop_ini(&ii) );
        BPC( bop_ini(&outii) );
    }

    read(a, /**/ cc, pp, ii);

    filter(a.color, cc, pp, ii, /**/ outpp, outii);

    write(a, outpp, outii);
    
    BPC( bop_fin(cc) );    
    BPC( bop_fin(pp) );
    BPC( bop_fin(outpp) );
    if (a.ii) {
        BPC( bop_fin(ii) );
        BPC( bop_fin(outii) );
    }
    
    return 0;
}

/*

  # TEST: color.t0
  # rm -f *out.txt
  # make 
  # t=out
  # ./color 0 $t data/test.bop data/colors.bop
  # bop2txt $t.pp.bop > pp.out.txt

*/


