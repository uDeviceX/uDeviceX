#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "bop_common.h"
#include "type.h"
#include "utils.h"
#include "macros.h"

namespace bop_utils {

static const char *typestr[] = {
    "float", "double", "int", "fascii", "iascii"};

BopStatus safe_malloc(size_t sz, void **data) {
    *data = malloc(sz);
    
    if (*data == NULL) {
        report_err("could not allocate array of %ld bytes\n", sz);
        return BOP_BADALLOC;
    }
    return BOP_SUCCESS;
}

BopStatus safe_open(const char *fname, const char *mode, FILE **f) {
    *f = fopen(fname, mode);
    if (*f == NULL) {
        report_err("could not open <%s>\n", fname);
        return BOP_BADFILE;
    }
    return BOP_SUCCESS;
}

size_t get_bsize(BopType t) {
    switch(t) {
    case BopFLOAT:
    case BopFASCII:
        return sizeof(float);
    case BopDOUBLE:
        return sizeof(double);
    case BopINT:
    case BopIASCII:
        return sizeof(int);
    };
    return 0;
}

#define SEP '/'
void get_path(const char *full, char *path) {
    int i = strlen(full);
    while (--i >= 0 && full[i] != SEP);

    if (i) memcpy(path, full, (i+1)*sizeof(char));
}

void get_fname_values(const char *fnbop, char *fnval) {
    int i = strlen(fnbop);
    const int n = i;
    while (--i >= 0 && fnbop[i] != SEP);

    memcpy(fnval, fnbop + i + 1, (n-i)*sizeof(char));

    i = strlen(fnval);
    strncpy(fnval + i - 4, ".values", 8);
}

#undef SEP

BopType str2type(const char *str) {
    if      (strcmp(str,  "float") == 0) return BopFLOAT;
    else if (strcmp(str, "double") == 0) return BopDOUBLE;
    else if (strcmp(str,    "int") == 0) return BopINT;
    else if (strcmp(str,  "ascii") == 0) return BopFASCII;
    else if (strcmp(str, "iascii") == 0) return BopIASCII;
    // default is float
    return BopFLOAT;
}

const char * type2str(BopType t) {
    if (t >= 0 && t < _BopNTYPES)
        return typestr[t];
    return NULL;
}

void report_err(const char *fmt, ...) {
    va_list ap;
    char msg[CERRSIZE];
    
    va_start(ap, fmt);
    vsnprintf(msg, CERRSIZE - 1, fmt, ap);
    va_end(ap);

    snprintf(bop_error_msg, CERRSIZE, ":%s:%d: %s",
             __FILE__, __LINE__, msg);
}

} // bop_header
