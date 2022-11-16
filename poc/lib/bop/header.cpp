#include <stdio.h>
#include <string.h>

#include "bop_common.h"
#include "type.h"
#include "utils.h"
#include "macros.h"

#include "header.h"

namespace bop_header {
using namespace bop_utils;

static int is_desc(const char *s, const char *d) {
    if (strcmp(s, d) == 0)
        return 1;
    return 0;
}

static void read_data_file(const char *data, /**/ char *df) {
    strcpy(df, data);
}

static void read_variables(const char *data, /**/ BopData *d) {
    strcpy(d->vars, data);
}

static void read_type(const char *data, /**/ BopData *d) {
    d->type = str2type(data);
}

static void extract_desc_data(const char *l, /**/ char *desc, char *data) {
    const char *c;
    char *d;
    for (c = l, d = desc; l < l + strlen(l); ++c, ++d) {
        if (*c == ':')
            break;
        *d = *c;
    }
    while (*c == ':' || *c == ' ') ++c;
    strcpy(data, c);
}

static void read_nrank(int rank, const char *data, /* io */ FILE *f, /**/ BopData *d) {
    int i, size;
    long nloc;
    sscanf(data, "%d", &size);
    for (i = 0; i < size; ++i) {
        fscanf(f, "%ld\n", &nloc);
        if (i == rank) d->n = nloc;
    }
}

static BopStatus parse_entry(int rank, const char *l, /**/ FILE *f, char *dfname, BopData *d) {
    char desc[CBUFSIZE] = {0}, data[CBUFSIZE] = {0};
    extract_desc_data(l, /**/ desc, data);
    
    if (is_desc(desc, "DATA_FILE"))
        read_data_file(data, /**/ dfname);
    else if (is_desc(desc, "DATA_FORMAT"))
        read_type(data, /**/ d);
    else if (is_desc(desc, "VARIABLES"))
        read_variables(data, /**/ d);
    else if (is_desc(desc, "NRANK"))
        read_nrank(rank, data, /**/ f, /**/ d);
    else {
        report_err("unprocessed desc: <%s>, data: <%s>\n", desc, data);
        return BOP_WFORMAT;
    }
    return BOP_SUCCESS;
}

static BopStatus read_n_data(const char *data, /**/ BopData *d) {
    long n;
    if (1 != sscanf(data, "%ld", &n))
        return BOP_WFORMAT;
    d->n = n;
    return BOP_SUCCESS;
}

static int count_vars(const char *var) {
    int i;
    char buf[CBUFSIZE];

    i = 0;
    while (1 == sscanf(var, "%s", buf)) {
        var = strstr(var, buf);
        var += strlen(buf);
        ++i;
    }
    return i;
}

static bool read_entry(FILE *f, char *l) {
    int ret = fscanf(f, " %" xstr(CBUFSIZE) "[^\n]c", l);
    return EOF != ret;
}

BopStatus read_header(int rank, const char *fname, /**/ char *dfname, BopData *d) {
    FILE *f;
    char line[CBUFSIZE];
    int l = 0;
    BopStatus s;
    
    s = safe_open(fname, "r", &f);
    if (s != BOP_SUCCESS) return s;
    
    while (read_entry(f, line)) {
        /* first entry must contain number of particles */
        if (l == 0) s = read_n_data(line, /**/ d);
        else        s = parse_entry(rank, line, /**/ f, dfname, d);

        if (s != BOP_SUCCESS) return s;
        ++l;
    }    

    d->nvars = count_vars(d->vars);
    
    fclose(f);
    return s;
}

} // bop_header
