#include <stdio.h>
#include <stdlib.h>

#include "imp.h"
#include "utils/error.h"

int emalloc(size_t size, /**/ void **data) {
    *data = malloc(size);

    if (NULL == *data) {
        ERR("Failed to allocate array of size %ld\n", size);
        return 1;
    }
    return 0;
}

void efree(void *ptr) { free(ptr); }

int efopen(const char *fname, const char *mode, /**/ FILE **f) {
    *f = fopen(fname, mode);

    if (NULL == *f) {
        ERR("Could not open file <%s> with mode <%s>", fname, mode);
        return 1;
    }
    return 0;
}

void efclose(FILE *f) {fclose(f);}
    
