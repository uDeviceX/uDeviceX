#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bop_common.h"
#include "bop_serial.h"
#include "check.h"

template <typename T>
T EndSwap(T f) {
    static_assert(sizeof(T) == 4 * sizeof(unsigned char), "wrong type: must have 4 bytes");
    union {
        T f;
        unsigned char b[4];
    } dat1, dat2;

    dat1.f = f;
    dat2.b[0] = dat1.b[3];
    dat2.b[1] = dat1.b[2];
    dat2.b[2] = dat1.b[1];
    dat2.b[3] = dat1.b[0];
    return dat2.f;
}

namespace vtk {
float *rr, *ff; /* positions, fields */
int *ii;        /* integer fields */
    
template <typename T>
void init(const long n, const int nvars, const T *data) {
    long i;
    int d, nf;
    if (nvars < 3) {
        fprintf(stderr, "Need at least 3 coordinates x y z\n");
        exit(1);
    }

    nf = nvars - 3;

    ff = rr = NULL;
        
    if (nf > 0)
    ff = new float[nf * n];
    rr = new float[3  * n];
        
    for (i = 0; i < n; ++i) {
        for (d = 0; d < 3; ++d)
        rr[3*i + d] = (float) data[nvars*i + d];

        for (d = 0; d < nf; ++d)
        ff[n*d + i] = (float) data[nvars*i + 3 + d];
    }

    for (i = 0; i < 3  * n; ++i) rr[i] = EndSwap(rr[i]);
    for (i = 0; i < nf * n; ++i) ff[i] = EndSwap(ff[i]);
}

void init_i(const long n, const int nvars, const int *data) {
    long i;
    int d;
    ii = NULL;
        
    if (nvars > 0)
    ii = new int[nvars * n];
        
    for (i = 0; i < n; ++i)
        for (d = 0; d < nvars; ++d)
            ii[nvars*d + i] = data[nvars*i + d];

    for (i = 0; i < nvars * n; ++i) ii[i] = EndSwap(ii[i]);
}

void finalize() {
    if (rr) delete[] rr;
    if (ff) delete[] ff;
    if (ii) delete[] ii;
}
    
void header(FILE *f, const long n) {
    fprintf(f, "# vtk DataFile Version 2.0\n");
    fprintf(f, "created with bop2vtk\n");
    fprintf(f, "BINARY\n");
}

void vertices(FILE *f, const long n) {
    fprintf(f, "DATASET POLYDATA\n");
    fprintf(f, "POINTS %ld float\n", n);
    fwrite(rr, 3*n, sizeof(float), f);
    fprintf(f, "\n");
}

void fields(FILE *f, const long n, const int nvars, const Cbuf *vars) {
    if (nvars <= 3) return;

    fprintf(f, "POINT_DATA %ld\n", n);

    for (int i = 3; i < nvars; ++i) {
        fprintf(f, "SCALARS %s float\n", vars[i].c);
        fprintf(f, "LOOKUP_TABLE default\n");
        fwrite(ff + (i-3)*n, n, sizeof(float), f);
    }
}

void ifields(FILE *f, const long n, const int nvars, const Cbuf *vars) {
    if (nvars <= 0) return;

    for (int i = 0; i < nvars; ++i) {
        fprintf(f, "SCALARS %s int\n", vars[i].c);
        fprintf(f, "LOOKUP_TABLE default\n");
        fwrite(ii + i * n, n, sizeof(int), f);
    }
}
} // vtk

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <out.vtk> <in1.bop> <in2.bop> ... -- <in1.int.bop> <in2.int.bop> ...\n", argv[0]);
        exit(1);
    }
    int i_int, i, ninput, nd;
    BopData **fdd, **idd, *d, *di;
    BopType type;
    int nvars, nivars;
    long n;
    char dfname[256];
    
    i_int = -1;
    for (i = 2; i < argc; ++i) if (strcmp(argv[i], "--") == 0) i_int = i + 1; 

    const bool read_int = i_int != -1;
    
    ninput = argc-2;
    nd = read_int ? (ninput - 1) / 2 : ninput;
    
    fdd = new BopData*[nd];
    idd = new BopData*[nd];

    BPC(bop_ini(&d));
    if (read_int) BPC(bop_ini(&di));
        
    for (i = 0; i < nd; ++i) {
        BPC(bop_ini(fdd + i));
        BPC(bop_read_header(argv[2+i], /**/ fdd[i], dfname));
        BPC(bop_alloc(fdd[i]));
        BPC(bop_read_values(dfname, /**/ fdd[i]));
        
        if (read_int) {
            BPC(bop_ini(idd + i));
            BPC(bop_read_header(argv[i_int+i], /**/ idd[i], dfname));
            BPC(bop_alloc(idd[i]));
            BPC(bop_read_values(dfname, /**/ idd[i]));
        }
    }

    BPC(bop_concatenate(nd, (const BopData**) fdd, /**/ d));
    if (read_int) BPC(bop_concatenate(nd, (const BopData**) idd, /**/ di));

    // BPC(bop_summary(d));
    // if (read_int) BPC(bop_summary(di));
        
    FILE *f = fopen(argv[1], "w");

    BPC(bop_get_type(d, &type));
    BPC(bop_get_n(d, &n));
    BPC(bop_get_nvars(d, &nvars));
    
    switch (type) {
    case BopFLOAT:
    case BopFASCII:
        vtk::init(n, nvars, (const float*) bop_get_data(d));
        break;
    case BopDOUBLE:
        vtk::init(n, nvars, (const double *) bop_get_data(d));
        break;
    case BopINT:
    case BopIASCII:
        break;
    };

    if (read_int) {
        BPC(bop_get_nvars(di, &nivars));
        vtk::init_i(n, nivars, (const int *) bop_get_data(di));
    }

    Cbuf *vars, *ivars;
    vars = ivars = NULL;
    
    vars = new Cbuf[nvars];
    BPC(bop_get_vars(d, /**/ vars));

    if (read_int) {
        ivars = new Cbuf[nivars];
        BPC(bop_get_vars(di, /**/ ivars));
    }
    
    vtk::header  (f, n);
    vtk::vertices(f, n);
    vtk::fields  (f, n, nvars, vars);
    if (read_int) vtk::ifields(f, n, nivars, ivars);
    vtk::finalize();

    fclose(f);

    for (i = 0; i < nd; ++i) {
        BPC(bop_fin(fdd[i]));
        if (read_int) BPC(bop_fin(idd[i]));
    }
    BPC(bop_fin(d));
    delete[] vars;

    if (read_int) {
        BPC(bop_fin(di));
        delete[] ivars;
    }
    
    delete[] fdd;
    delete[] idd;
    
    return 0;
}

/*

  # nTEST: bop2vtk.t0
  # make 
  # ./bop2vtk test.out.vtk data/test.bop

*/
