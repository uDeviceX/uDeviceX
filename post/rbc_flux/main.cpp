#include <stdio.h>
#include <stdlib.h>

#include "type.h"
#include "com.h"

struct Arg {
    int Lx, Ly, Lz;
    int nfiles;
    char **fnames;
};

static void usg() {
    fprintf(stderr, "usage: u.post.rbc.flux <Lx> <Ly> <Lz> <com-00.txt> <com-01.txt> ...\n");
    exit(1);
}

static bool shift(int *c, char ***v) {
    (*c) --;
    (*v) ++;
    return *c > 0;
}

static void parse(int c, char **v, Arg *a) {
    if (!shift(&c, &v)) usg();
    a->Lx = atof(*v);
    if (!shift(&c, &v)) usg();
    a->Ly = atof(*v);
    if (!shift(&c, &v)) usg();
    a->Lz = atof(*v);

    if (!shift(&c, &v)) usg();
    a->fnames = v;
    a->nfiles = c;

    if (c < 2) usg();        
}

template <typename T>
static void swap(T *a, T *b) {
    T c = *a;
    *a = *b;
    *b = c;
}

int main(int argc, char **argv ) {
    Arg a;
    Com *cc0, *cc1;
    int i, n;
    FILE *f;

    parse(argc, argv, &a);


    f = fopen(a.fnames[0], "r");
    read(f, &n, &cc0);
    sort_by_id(n, cc0);
    fclose(f);
    
    for (i = 1; i < a.nfiles; ++i) {
        f = fopen(a.fnames[i], "r");
        read(f, &n, &cc1);
        fclose(f);
        
        sort_by_id(n, cc1);

        swap(&cc0, &cc1);
        
        free(cc1);
    }
    free(cc0);
    
    return 0;
}
