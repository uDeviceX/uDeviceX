#include <stdio.h>
#include "utils/imp.h"

#include "type.h"
#include "imp.h"

void ini(Array3d **pq) {
    Array3d *q;
    emalloc(sizeof(Array3d), /**/ (void**)&q);
    *pq = q;
}

void fin(Array3d *q) {
    efree(q);
}
