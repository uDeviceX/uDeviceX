#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "bop_common.h"
#include "type.h"
#include "macros.h"
#include "utils.h"

using namespace bop_utils;

char bop_error_msg[1024] = {0};
static const char * err_desc[_BOP_NERR] = {
    "success",
    "bad allocation",
    "bad file descriptor",
    "wrong number of variables",
    "file types mismatch",
    "wrong format",
    "wrong mpi size",
    "null pointer",
    "overflow"
};

BopStatus bop_ini(BopData **d) {
    BopStatus s;
    BopData *b;
    s = safe_malloc(sizeof(BopData), (void **) d);
    if (s == BOP_SUCCESS) {
        b = *d;
        b->n = 0;
        b->nvars = 0;
        b->vars[0] = '\0';
        b->type = BopFLOAT;
        b->data = NULL;
    }
    return s;
}

BopStatus bop_alloc(BopData *d) {
    size_t sz, bsize;
    bsize = get_bsize(d->type);
    sz = d->n * d->nvars * bsize;

    return safe_malloc(sz, &d->data);
}

BopStatus bop_fin(BopData *d) {
    if (d->data) free(d->data);
    free(d);
    return BOP_SUCCESS;
}

BopStatus bop_set_n(long n, BopData *d) {
    if (d) {
        d->n = n;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}
        
BopStatus bop_set_vars(int n, const char *vars, BopData *d) {
    if (d) {
        d->nvars = n;
        strcpy(d->vars, vars);
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

BopStatus bop_set_type(BopType type, BopData *d) {
    if (d) {
        d->type = type;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

BopStatus bop_get_n(const BopData *d, long *n) {
   if (d) {
        *n = d->n;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

BopStatus bop_get_nvars(const BopData *d, int *n) {
   if (d) {
        *n = d->nvars;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

static BopStatus extract_vars(int n, const char *v, /**/ Cbuf *vars) {
    int i, ret;
    for (i = 0; i < n; ++i) {
        ret = sscanf(v, "%s", vars[i].c);
        if (ret != 1)
            return BOP_WRONGVAR;
        v = strstr(v, vars[i].c);
        v += strlen(vars[i].c);
    }
    return BOP_SUCCESS;
}

BopStatus bop_get_vars(const BopData *d, Cbuf *vars) {
   if (d) {
        return extract_vars(d->nvars, d->vars, /**/ vars);
    }
    return BOP_NULLPTR;
}

BopStatus bop_get_vars(const BopData *d, const char **vars) {
   if (d) {
       *vars = d->vars;
       return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

BopStatus bop_get_type(const BopData *d, BopType *type) {
   if (d) {
        *type = d->type;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

void* bop_get_data(BopData *d) {
    return d->data;
}

const void* bop_get_data(const BopData *d) {
    return d->data;
}


BopStatus bop_summary(const BopData *d) {
    fprintf(stderr, "(reader) found %ld entries, %d field(s)\n", d->n, d->nvars);
    fprintf(stderr, "\tformat: %s\n", type2str(d->type));
    fprintf(stderr, "\tvars: %s\n", d->vars);
    return BOP_SUCCESS;
}

BopStatus bop_concatenate(const int nd, const BopData **dd, BopData *dall) {
    const BopData *d = dd[0];
    long n             = d->n;
    const BopType type = d->type;
    const int nvars    = d->nvars;
    size_t bsize;
    int i;
    long ni, start;
    const void *src;
    char *dst;
    
    for (i = 1; i < nd; ++i) {
        d = dd[i];
        n += d->n;
        if (type != d->type || nvars != d->nvars)
            return BOP_MISMATCH;
    }
    bsize = get_bsize(type);

    dall->n = n;
    dall->nvars = nvars;
    dall->type = type;
    bop_alloc(dall);

    d = dd[0];
    strcpy(dall->vars, d->vars);
    
    start = 0;
    
    for (i = 0; i < nd; ++i) {
        d = dd[i];
        ni = d->n;
        src = d->data;
        dst = (char *) dall->data + bsize * start;
        memcpy(dst, src, ni * nvars * bsize);
        
        start += ni * nvars;
    }

    return BOP_SUCCESS;
}

bool bop_success(BopStatus status) {
    return status == BOP_SUCCESS;
}

const char* bop_report_error_desc(BopStatus status) {
    assert(status >= 0 && status < _BOP_NERR);
    return err_desc[status];
}

char* bop_report_error_mesg() {
    return bop_error_msg;
}

