#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "type.h"
#include "com.h"

static int count_lines(FILE *f) {
    int l = 0;
    char c;
    while(!feof(f)) {
        c = fgetc(f);
        if (c == '\n') l++;
    }
    return l;
}

static void read_entry(FILE *f, Com *c) {
    fscanf(f, "%d %f %f %f %f %f %f\n",
           &c->id,
           &c->x, &c->y, &c->z,
           &c->vx, &c->vy, &c->vz);
}

void read(FILE *f, int *n_, Com **cc_) {
    int i, n;
    Com *cc;
    n = count_lines(f);
    rewind(f);
    cc = (Com*) malloc(n * sizeof(Com));

    for (i = 0; i < n; ++i)
        read_entry(f, &cc[i]);

    *cc_ = cc;
    *n_ = n;
}

static bool compare(Com a, Com b) {return a.id < b.id;}

void sort_by_id(int n, Com *cc) {
    std::sort(cc, cc+n, &compare);
}
