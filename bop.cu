#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include "config.cuh"
#include "constants.cuh"
#include "types.cuh"
#include "device.cuh"

#ifndef UDX_BOP_COMMON
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

struct Cbuf {
    enum {SIZ=256};
    char c[SIZ];
};
struct BopData;
typedef int BopStatus;

enum {BopFLOAT, BopDOUBLE, BopINT, BopFASCII, BopIASCII, _BopNTYPES};
typedef int BopType;

BopStatus bop_ini(BopData **d);
BopStatus bop_alloc(BopData *d);
BopStatus bop_fin(BopData *d);

BopStatus bop_set_n(long n, BopData *d);
BopStatus bop_set_vars(int n, const char *vars, BopData *d);
BopStatus bop_set_type(BopType type, BopData *d);
void* bop_get_data(BopData *d);

BopStatus bop_get_n(const BopData *d, long *n);
BopStatus bop_get_nvars(const BopData *d, int *n);
BopStatus bop_get_vars(const BopData *d, Cbuf *vars);
BopStatus bop_get_vars(const BopData *d, const char **vars);
BopStatus bop_get_type(const BopData *d, BopType *type);
const void* bop_get_data(const BopData *d);

BopStatus bop_summary(const BopData *d);
BopStatus bop_concatenate(const int nd, const BopData **dd, BopData *dall);

bool         bop_success(BopStatus status);
const char * bop_report_error_desc(BopStatus status);
char *       bop_report_error_mesg();

#define CBUFSIZE 256
#define CERRSIZE 1024

struct BopData {
    long n;
    int nvars;
    char vars[CBUFSIZE];
    BopType type;
    void *data;
};

extern char bop_error_msg[CERRSIZE];

enum {
    BOP_SUCCESS,
    BOP_BADALLOC,
    BOP_BADFILE,
    BOP_WRONGVAR,
    BOP_MISMATCH,
    BOP_WFORMAT,
    BOP_NULLPTR,
    BOP_WMPISIZE,
    BOP_OVERFLOW,
    _BOP_NERR
};

#define xstr(s) str(s)
#define str(s) #s

BopStatus safe_malloc(size_t sz, void **data);
BopStatus safe_open(const char *fname, const char *mode, FILE **f);

size_t get_bsize(BopType t);

void get_path(const char *full, char *path);
void get_fname_values(const char *fnbop, char *fnval);

const char * type2str(BopType t);
BopType str2type(const char *str);

void report_err(const char *fmt, ...);

enum {SERIAL = -1};
BopStatus read_header(int rank, const char *fname,      char *dfname, BopData *d);

BopStatus bop_write_header(MPI_Comm comm, const char *name, const BopData *d);
BopStatus bop_write_values(MPI_Comm comm, const char *name, const BopData *d);

BopStatus bop_read_header(MPI_Comm comm, const char *hfname, BopData *d, char *dfname);
BopStatus bop_read_values(MPI_Comm comm, const char *dfname, BopData *d);

BopStatus bop_write_header(const char *name, const BopData *d);
BopStatus bop_write_values(const char *name, const BopData *d);

BopStatus bop_read_header(const char *hfname, BopData *d, char *dfname);
BopStatus bop_read_values(const char *dfname, BopData *d);

#define UDX_BOP_COMMON
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

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

static BopStatus extract_vars(int n, const char *v,      Cbuf *vars) {
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
        return extract_vars(d->nvars, d->vars,      vars);
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

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

#include <stdio.h>
#include <string.h>

static int is_desc(const char *s, const char *d) {
    if (strcmp(s, d) == 0)
        return 1;
    return 0;
}

static void read_data_file(const char *data,      char *df) {
    strcpy(df, data);
}

static void read_variables(const char *data,      BopData *d) {
    strcpy(d->vars, data);
}

static void read_type(const char *data,      BopData *d) {
    d->type = str2type(data);
}

static void extract_desc_data(const char *l,      char *desc, char *data) {
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

static void read_nrank(int rank, const char *data,          FILE *f,      BopData *d) {
    int i, size;
    long nloc;
    sscanf(data, "%d", &size);
    for (i = 0; i < size; ++i) {
        fscanf(f, "%ld\n", &nloc);
        if (i == rank) d->n = nloc;
    }
}

static BopStatus parse_entry(int rank, const char *l,      FILE *f, char *dfname, BopData *d) {
    char desc[CBUFSIZE] = {0}, data[CBUFSIZE] = {0};
    extract_desc_data(l,      desc, data);

    if (is_desc(desc, "DATA_FILE"))
        read_data_file(data,      dfname);
    else if (is_desc(desc, "DATA_FORMAT"))
        read_type(data,      d);
    else if (is_desc(desc, "VARIABLES"))
        read_variables(data,      d);
    else if (is_desc(desc, "NRANK"))
        read_nrank(rank, data,      f,      d);
    else {
        report_err("unprocessed desc: <%s>, data: <%s>\n", desc, data);
        return BOP_WFORMAT;
    }
    return BOP_SUCCESS;
}

static BopStatus read_n_data(const char *data,      BopData *d) {
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

BopStatus read_header(int rank, const char *fname,      char *dfname, BopData *d) {
    FILE *f;
    char line[CBUFSIZE];
    int l = 0;
    BopStatus s;

    s = safe_open(fname, "r", &f);
    if (s != BOP_SUCCESS) return s;

    while (read_entry(f, line)) {

        if (l == 0) s = read_n_data(line,      d);
        else        s = parse_entry(rank, line,      f, dfname, d);

        if (s != BOP_SUCCESS) return s;
        ++l;
    }

    d->nvars = count_vars(d->vars);

    fclose(f);
    return s;
}

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

template <typename T>
static void write_mpi(MPI_Comm comm, const char *fname, long n, const T *data, MPI_Datatype type) {
    MPI_Offset base, offset, len;
    MPI_Status status;
    MPI_File f;

    MPI_File_open(comm, fname , MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f);
    MPI_File_set_size(f, 0);
    MPI_File_get_position(f, &base);

    len = n * sizeof(T);
    offset = 0;
    MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
    MPI_File_write_at_all(f, base + offset, data, n, type, &status);
    MPI_File_close(&f);
}

static BopStatus write_header(MPI_Comm comm, const char *fhname, const char *fdname, const BopData *d) {
    enum {
        ROOT     = 0,
        MAX_SIZE = 2048
    };
    int rank, size, nchar;
    long nloc, ntot;
    char *buf;
    BopStatus s;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    nloc = d->n;
    ntot = 0;
    MPI_Reduce(&nloc, &ntot, 1, MPI_LONG, MPI_SUM, ROOT, comm);

    s = safe_malloc(MAX_SIZE * sizeof(char), (void**) &buf);
    if (s != BOP_SUCCESS) return s;

    nchar = 0;
    if (rank == ROOT) {
        nchar += sprintf(buf + nchar, "%ld\n", ntot);
        nchar += sprintf(buf + nchar, "DATA_FILE: %s\n", fdname);
        nchar += sprintf(buf + nchar, "DATA_FORMAT: %s\n", type2str(d->type));
        nchar += sprintf(buf + nchar, "VARIABLES: %s\n", d->vars);
        nchar += sprintf(buf + nchar, "NRANK: %d\n", size);
    }
    nchar += sprintf(buf + nchar, "%ld\n", d->n);

    if (nchar >= MAX_SIZE) {
        report_err("buffer is too small for header: %d, %d\n", nchar, MAX_SIZE);
        return BOP_OVERFLOW;
    }

    write_mpi(comm, fhname, nchar, buf, MPI_CHAR);

    free(buf);
    return s;
}

BopStatus bop_write_header(MPI_Comm comm, const char *name, const BopData *d) {
    char fnval[CBUFSIZE] = {0},
        fnval0[CBUFSIZE] = {0},
        fnhead[CBUFSIZE] = {0};

    sprintf(fnhead, "%s.bop", name);

    get_path(fnhead, fnval);
    get_fname_values(fnhead, fnval0);
    strcat(fnval, fnval0);

    return write_header(comm, fnhead, fnval0, d);
}

template <typename T>
static BopStatus write_ascii(MPI_Comm comm, const char *pattern, const char *fname, const T *data, long n, int nvars) {
    char *buf;
    long i, k, j = 0, ns = 0;
    BopStatus s;
    enum {NPN = 16};

    s = safe_malloc(NPN * n * nvars * sizeof(char), (void**) &buf);
    if (s != BOP_SUCCESS) return s;

    for (i = 0; i < n; ++i) {
        for (k = 0; k < nvars; ++k)
            ns += sprintf(buf + ns, pattern, data[j++]);
        ns += sprintf(buf + ns, "\n");
    }
    write_mpi(comm, fname, ns, buf, MPI_CHAR);
    free(buf);
    return s;
}

static BopStatus write_data(MPI_Comm comm, const char *fnval, const BopData *d) {
    long n = d->n;
    int nvars = d->nvars;

    switch(d->type) {
    case BopFLOAT:
        write_mpi(comm, fnval, n * nvars, (const float *) d->data, MPI_FLOAT);
        break;
    case BopDOUBLE:
        write_mpi(comm, fnval, n * nvars, (const double *) d->data, MPI_DOUBLE);
        break;
    case BopINT:
        write_mpi(comm, fnval, n * nvars, (const int *) d->data, MPI_INT);
        break;
    case BopFASCII:
        return write_ascii(comm, "%.6e", fnval, (const float *) d->data, n, nvars);
        break;
    case BopIASCII:
        return write_ascii(comm, "%d", fnval, (const int *) d->data, n, nvars);
        break;
    }
    return BOP_SUCCESS;
}

BopStatus bop_write_values(MPI_Comm comm, const char *name, const BopData *d) {
    char dfname[CBUFSIZE] = {0};
    sprintf(dfname, "%s.values", name);
    return write_data(comm, dfname, d);
}

BopStatus bop_read_header(MPI_Comm comm, const char *hfname, BopData *d, char *dfname) {
    BopStatus s;
    char dfname0[CBUFSIZE] = {0}, locdfname[CBUFSIZE] = {0};
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    s = read_header(rank, hfname,      dfname0, d);

    get_path(hfname, locdfname);
    strcat(locdfname, dfname0);
    strcpy(dfname, locdfname);

    return s;
}

template <typename T>
static BopStatus read_mpi(MPI_Comm comm, const char *fname, long n, MPI_Datatype type, T *data) {
    MPI_Offset base, offset, len;
    MPI_Status status;
    MPI_File f;

    MPI_File_open(comm, fname , MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
    MPI_File_get_position(f, &base);

    len = n * sizeof(T);
    offset = 0;
    MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
    MPI_File_read_at_all(f, base + offset, data, n, type, &status);
    MPI_File_close(&f);
    return BOP_SUCCESS;
}

BopStatus bop_read_values(MPI_Comm comm, const char *dfname, BopData *d) {
    long n = d->n;
    int nvars = d->nvars;

    switch (d->type) {
    case BopFLOAT:
        return read_mpi(comm, dfname, n * nvars, MPI_FLOAT, (float *) d->data);
    case BopDOUBLE:
        return read_mpi(comm, dfname, n * nvars, MPI_DOUBLE, (double *) d->data);
    case BopINT:
        return read_mpi(comm, dfname, n * nvars, MPI_INT, (int *) d->data);
    case BopFASCII:
    case BopIASCII:
        fprintf(stderr, "Not implemented\n");
        return BOP_WFORMAT;
    };
    return BOP_SUCCESS;
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static BopStatus write_header(const char *fhname, const char *fdname, const BopData *d) {
    FILE *f;
    BopStatus s;

    s = safe_open(fhname, "w", &f);
    if (s != BOP_SUCCESS) return s;

    fprintf(f, "%ld\n", d->n);
    fprintf(f, "DATA_FILE: %s\n", fdname);
    fprintf(f, "DATA_FORMAT: %s\n", type2str(d->type));
    fprintf(f, "VARIABLES: %s\n", d->vars);

    fclose(f);
    return s;
}

BopStatus bop_write_header(const char *name, const BopData *d) {
    char fnval[CBUFSIZE] = {0},
        fnval0[CBUFSIZE] = {0},
        fnhead[CBUFSIZE] = {0};

    sprintf(fnhead, "%s.bop", name);

    get_path(fnhead, fnval);
    get_fname_values(fnhead, fnval0);
    strcat(fnval, fnval0);

    return write_header(fnhead, fnval0, d);
}

template <typename T>
static void write_ascii(const char pattern[], const T *data, const long n, const long nvars, FILE *f) {
    long i, k, j = 0;
    for (i = 0; i < n; ++i) {
        for (k = 0; k < nvars; ++k)
            fprintf(f, pattern, data[j++]);
        fprintf(f, "\n");
    }
}

static BopStatus write_data(const char *fnval, const BopData *d) {
    FILE *fd;
    size_t bsize;
    BopStatus s;

    s = safe_open(fnval, "w", &fd);
    bsize = get_bsize(d->type);

    if (s != BOP_SUCCESS) return s;

    switch(d->type) {
    case BopFLOAT:
    case BopDOUBLE:
    case BopINT:
        fwrite(d->data, bsize, d->n * d->nvars, fd);
        break;
    case BopFASCII:
        write_ascii("%.6e ", (const float*) d->data, d->n, d->nvars, fd);
        break;
    case BopIASCII:
        write_ascii("%d ", (const int*) d->data, d->n, d->nvars, fd);
        break;
    }
    fclose(fd);

    return s;
}

BopStatus bop_write_values(const char *name, const BopData *d) {
    char dfname[CBUFSIZE] = {0};
    sprintf(dfname, "%s.values", name);
    return write_data(dfname, d);
}

BopStatus bop_read_header(const char *hfname, BopData *d, char *dfname) {
    BopStatus s;
    char dfname0[CBUFSIZE] = {0}, locdfname[CBUFSIZE] = {0};
    s = read_header(SERIAL, hfname,      dfname0, d);

    get_path(hfname, locdfname);
    strcat(locdfname, dfname0);
    strcpy(dfname, locdfname);

    return s;
}

static BopStatus read_values(const char *fn, long n, int nvars, size_t bsize, void *data) {
    FILE *f;
    BopStatus s;
    s = safe_open(fn, "r", &f);
    if (s != BOP_SUCCESS) return s;

    fread(data, bsize, n * nvars, f);

    fclose(f);
    return s;
}

template <typename real>
static BopStatus read_ascii_values(const char pattern[], const char *fn, void *data) {
    char buf[CBUFSIZE] = {0}, *str;
    long i = 0, j;
    FILE *f;
    BopStatus s;
    real *d, a;

    s = safe_open(fn, "r", &f);
    if (s != BOP_SUCCESS) return s;

    d = (real*) data;

    while (fscanf(f, " %[^\n]" xstr(MAXC) "c", buf) == 1) {
        str = buf;
        j = 0;
        while (sscanf(str, pattern, &a, &j) == 1) {
            str += j;
            d[i++] = a;
        }
        memset(buf, 0,  sizeof(buf));
    }

    fclose(f);
    return s;
}

BopStatus bop_read_values(const char *dfname, BopData *d) {
    size_t bsize;
    bsize = get_bsize(d->type);

    switch (d->type) {
    case BopFLOAT:
    case BopDOUBLE:
    case BopINT:
        return read_values(dfname, d->n, d->nvars, bsize, d->data);
    case BopFASCII:
        return read_ascii_values<float>(" %f%n", dfname, d->data);
    case BopIASCII:
        return read_ascii_values<int>  (" %d%n", dfname, d->data);
    };
    return BOP_SUCCESS;
}

