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
    a->pp = *argv;
    if (!arg_shift(&argc, &argv)) usg();
    a->cc = *argv;
    if (arg_shift(&argc, &argv))
        a->ii = *argv;
}

int main(int argc, char **argv) {
    Args a;
    BopData *pp, *cc, *ii;
    
    parse(argc, argv, /**/ &a);

    return 0;
}
