#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "he/err.h"
#include "he/memory.h"

int he_malloc(int size, /**/ void **pq) {
    void *q;
    q = malloc(size);
    if (q == NULL) ERR(HE_MEMORY, "malloc failed for size=%ld", size);
    *pq = q;
    return HE_OK;
}

int he_free(void *q) {
    free(q);
    q = NULL;
    return HE_OK;
}

int he_memcpy(void *dest, const void *src, size_t n) {
    if (dest == NULL) ERR(HE_MEMORY, "dest == NULL");
    if (src == NULL) ERR(HE_MEMORY, "src == NULL");
    memcpy(dest, src, n);
    return HE_OK;
}
