#include <stdio.h>
#include "utils/imp.h"

#include "type.h"
#include "imp.h"

#include "utils/error.h"

void ini(Array3d **pq) {
    Array3d *q;
    UC(emalloc(sizeof(Array3d), /**/ (void**)&q));
    *pq = q;
}

void fin(Array3d *q) {
    UC(efree(q));
}
