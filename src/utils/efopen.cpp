#include <stdio.h>

#include "efopen.h"
#include "utils/error.h"

int efopen(const char *fname, const char *mode, /**/ FILE **f) {
    *f = fopen(fname, mode);

    if (NULL == *f) {
        ERR("Could not open file <%s> with mode <%s>", fname, mode);
        return 1;
    }
    return 0;
}

void efclose(FILE *f) {fclose(f);}
    
