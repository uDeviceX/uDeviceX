#include <stdlib.h>

#include "halloc.h"
#include "utils/error.h"

int emalloc(size_t size, /**/ void **data) {
    *data = malloc(size);

    if (NULL == *data) {
        UERR("Failed to allocate array of size %ld\n", size);
        return 1;
    }
    return 0;
}

void efree(void *ptr) { free(ptr); }
