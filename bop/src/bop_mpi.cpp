#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bop_common.h"
#include "type.h"
#include "header.h"
#include "macros.h"
#include "utils.h"

#include "bop_mpi.h"

using namespace bop_header;
using namespace bop_utils;

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
    
    s = read_header(rank, hfname, /**/ dfname0, d);
    
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

template <typename T>
static BopStatus read_ascii() {
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
