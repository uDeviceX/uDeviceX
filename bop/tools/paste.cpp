#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bop_common.h"
#include "bop_serial.h"

#define ERR(...) do {                           \
        fprintf(stderr, __VA_ARGS__);           \
        exit(1);                                \
    } while (0);

#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

struct Args {
    char **in;
    char *out;
    int n;    
};

static void usg() {
    ERR("usage: bop.paste <out> <f1.bop> <f2.bop> ... \n");
}

static int shift_args(int *c, char ***v) {
    (*c)--;
    (*v)++;
    return (*c) > 0;
}

static void parse(int argc, char **argv, Args *a) {
    if (!shift_args(&argc, &argv)) usg();
    a->out = *argv;

    if (!shift_args(&argc, &argv)) usg();
    a->in = argv;
    a->n = argc;
    if (a->n < 2)
        ERR("given only %d files, expected at least 2\n", a->n);
}

static void read(const char *fh, BopData *b) {
    char fd[FILENAME_MAX] = {0};
    BPC(bop_read_header(fh, b, fd));
    BPC(bop_alloc(b));
    BPC(bop_read_values(fd, b));
}

static void write(const char *base, const BopData *b) {
    BPC(bop_write_header(base, b));
    BPC(bop_write_values(base, b));
}

template <typename T>
static void append(long n, int nvarsin, const T *in, int nvarsout, int startout, T *out) {
    long i;
    T *dst;
    const T *src;
    for (i = 0; i < n; ++i) {
        dst = out + i * nvarsout + startout;
        src = in  + i * nvarsin;
        memcpy(dst, src, nvarsin * sizeof(T));
    }
}

template <typename T>
static void paste(long n, int nvars, int nin, BopData **in, BopData *out) {
    long i;
    int start = 0, nvars0;
    const T *data_in;
    T *data_out = (T*) bop_get_data(out);

    for (i = 0; i < nin; ++i) {
        BPC(bop_get_nvars(in[i], &nvars0));
        data_in = (const T*) bop_get_data(in[i]);
        append(n, nvars0, data_in, nvars, start, data_out);
        start += nvars0;
    }    
}


int main(int argc, char **argv) {
    BopData **in, *out;
    Args a;
    int i;
    long n = 0, ncheck;
    int nvars, nvars0;
    char varsall[2048] = {0};
    const char *vars;
    BopType type = BopFLOAT, typecheck;
    
    parse(argc, argv, &a);

    in = (BopData**) malloc(a.n * sizeof(BopData*));

    BPC(bop_ini(&out));

    nvars = 0;
    for (i = 0; i < a.n; ++i) {
        BPC(bop_ini(in + i));
        read(a.in[i], in[i]);
        BPC(bop_get_n    (in[i], &ncheck));
        BPC(bop_get_nvars(in[i], &nvars0));
        BPC(bop_get_vars (in[i], &vars));
        BPC(bop_get_type (in[i], &typecheck));
        
        if (0 == i) {
            n = ncheck;
            type = typecheck;
        }
        else {
            if (ncheck != n)
                ERR("files must have the same number of rows (file <%s> has %ld)\n", a.in[i], ncheck);
            if (typecheck != type)
                ERR("files must have the same type (see file <%s>)\n", a.in[i]);
        }
        
        nvars += nvars0;
        strcat(varsall, vars);
        strcat(varsall, " ");
    }

    BPC(bop_set_n    (n,              out));
    BPC(bop_set_vars (nvars, varsall, out));
    BPC(bop_set_type (type,           out));
    
    BPC(bop_alloc(out));

    switch (type) {
    case BopFLOAT:
    case BopFASCII:
        paste<float>(n, nvars, a.n, in, out);
        break;
    case BopDOUBLE:
        paste<double>(n, nvars, a.n, in, out);
        break;
    case BopINT:
    case BopIASCII:
        paste<int>(n, nvars, a.n, in, out);
        break;        
    };

    write(a.out, out);

    for (i = 0; i < a.n; ++i)
        BPC(bop_fin(in[i]));

    BPC(bop_fin(out));

    free(in);

    return 0;
}

/*

# TEST: fascii.t0
make 
set -eu
./paste ascii data/ascii-?.bop
cat ascii.values > ascii.out.txt 

# TEST: iascii.t0
make 
set -eu
./paste iascii data/iascii-?.bop
cat iascii.values > iascii.out.txt 

*/
