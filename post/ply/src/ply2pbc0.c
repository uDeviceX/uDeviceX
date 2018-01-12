#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "endian.h"
#include "util.h"

float X, Y, Z;
int   nv; /* number of vertices */
int   nf; /* number of faces */

#define NVAR  6 /* x, y, z, vx, vy, vz */
#define NV_PER_FACE 3

float *buf;
float  *xx,  *yy,  *zz;
float *vvx, *vvy, *vvz;

/* vertex indexes in one face */
int *ff1, *ff2, *ff3;
int *ibuf;

FILE* fd_nl;
char line[BUFSIZ];
void init_nl(FILE* fd) {fd_nl = fd;}
void nl() { /* read [n]ext [l]ine; sets `line' */
    fgets(line, sizeof(line), fd_nl);
}

void read_header(FILE* fd) {
    init_nl(fd);
    nl(); /* ply */
    nl(); /* format binary_little_endian 1.0 */
    /* element vertex %nv% */
    nl(); sscanf(line, "element vertex %d\n", &nv);
    nl(); nl(); nl(); nl(); nl(); nl(); /* property float [xyzuvw] */

    nl(); sscanf(line, "element face %d\n", &nf);
    nl(); /* property list int int vertex_index */
    nl(); /* end_header */
}

void read_vertices(FILE* fd) {
    safe_fread(buf, NVAR*nv, sizeof(float), fd);
    for (int iv=0, ib=0; iv<nv; ++iv) {
        xx[iv] = buf[ib++];  yy[iv] = buf[ib++];  zz[iv] = buf[ib++];
        vvx[iv] = buf[ib++]; vvy[iv] = buf[ib++]; vvz[iv] = buf[ib++];
    }
}

void read_faces(FILE* fd) {
    safe_fread(ibuf, nf*(NV_PER_FACE+1), sizeof(int), fd);
    for (int ifa = 0, ib = 0; ifa < nf; ++ifa) {
        ib++; /* skip number of vertices per face */
        ff1[ifa] = ibuf[ib++]; ff2[ifa] = ibuf[ib++]; ff3[ifa] = ibuf[ib++];
    }
}

void* ealloc(size_t sz) {
    void *p;
    p = malloc(sz);
    if (p == NULL) {
        fprintf(stderr, "fail to alloc: %ld\n", sz);
        exit(-2);
    }
    return p;
}

void balloc() {
    buf = ealloc(NVAR*nv*sizeof(buf[0]));
    xx = ealloc(nv*sizeof(xx[0]));
    yy = ealloc(nv*sizeof(yy[0]));
    zz = ealloc(nv*sizeof(yy[0]));

    vvx = ealloc(nv*sizeof(vvx[0]));
    vvy = ealloc(nv*sizeof(vvy[0]));
    vvz = ealloc(nv*sizeof(vvz[0]));

    ff1 = ealloc(nf*sizeof(ff1[0]));
    ff2 = ealloc(nf*sizeof(ff2[0]));
    ff3 = ealloc(nf*sizeof(ff3[0]));

    ibuf = ealloc(nf*(NV_PER_FACE + 1)*sizeof(ibuf[0]));
}

#define pr(...) fprintf (fd, __VA_ARGS__)
void write_file_version(FILE* fd) {
    pr("# vtk DataFile Version 2.0\n");
}

void write_header(FILE* fd) {
    pr("Created with vrbc utils\n");
}

void write_format(FILE* fd) {
    pr("BINARY\n"); /* ASCII, BINARY */
}

#define SF(f) buf[ib++] = FloatSwap((f));
void write_vertices(FILE* fd) {
    const char* const dataType = "float";
    pr("DATASET POLYDATA\n");
    pr("POINTS %d %s\n", nv, dataType);

    int ib, iv;
    for (iv=0, ib=0; iv<nv; ++iv) {
        SF(xx[iv]); SF(yy[iv]); SF(zz[iv]);
    }
    fwrite(buf, ib, sizeof(float), fd);
}

#define SI(f) ibuf[ib++] = LongSwap(f);
void write_cells(FILE* fd) {
    int size = (1 + NV_PER_FACE)*nf;
    pr("\n");
    pr("POLYGONS %d %d\n", nf, size);
    int ib = 0;
    for (int f1, f2, f3, ifa = 0; ifa < nf; ++ifa) {
        f1 = ff1[ifa]; f2 = ff2[ifa]; f3 = ff3[ifa];
        SI(NV_PER_FACE); SI(f1); SI(f2); SI(f3);
    }
    fwrite(ibuf, ib, sizeof(int), fd);
}

void write_cells_attributes(FILE* fd) {
    int ifa, ib;
    const char* const dataType = "float";
    const char* const dataName = "s";

    pr("\n");
    pr("CELL_DATA %d\n", nf);
    pr("SCALARS %s %s\n", dataName, dataType);
    pr("LOOKUP_TABLE default\n");
    for (ifa = ib = 0; ifa < nf; ++ifa) SF(ifa);
    fwrite(buf, ib, sizeof(float), fd);
}

void write_file0(const char* fn) {
    FILE* fd = safe_fopen(fn, "w");
    write_file_version(fd);
    write_header(fd);
    write_format(fd);
    write_vertices(fd);
    write_cells(fd);
    write_cells_attributes(fd);
    fclose(fd);
}

void write_file(const char *d) {
    static int i = 0;
    char fn[BUFSIZ];
    sprintf(fn, "%s/%05d.vtk", d, i++);
    write_file0(fn);
}

void read_fst(const char *fn) {
    /* read and alloc */
    FILE *fd;
    fd = safe_fopen(fn, "r");
    read_header(fd);
    balloc();
    read_vertices(fd);
    read_faces(fd);
    fclose(fd);
}

void read_rst(const char *fn) {
    /* just read */
    FILE *fd;
    fd = safe_fopen(fn, "r");
    read_header(fd);
    read_vertices(fd);
    read_faces(fd);
    fclose(fd);
}

void compute_cm(/**/ float *px, float *py, float *pz) {
    float x, y, z;
    int iv;
    x = y = z = 0;
    for (iv = 0; iv < nv; iv++) {
        x += xx[iv]; y += yy[iv]; z += zz[iv];
    }
    x /= nv; y /= nv; z /= nv;
    *px = x; *py = y; *pz = z;
}

void shift(int ix, int iy, int iz) {
    int iv;
    if (ix == 0 && iy == 0 && iz == 0) return;
    for (iv = 0; iv < nv; iv++) {
        xx[iv] += ix*X;
        yy[iv] += iy*Y;
        zz[iv] += iz*Z;
    }
}

void upd_img0(float r, float r0, int *i) {
    if       (r - r0 >  X/2) (*i)--;
    else if  (r - r0 < -X/2) (*i)++;
}

void upd_img(float x,  float  y, float  z,
             float x0, float y0, float z0,
             int *ix, int *iy, int *iz) {
    upd_img0(x, x0, ix);
    upd_img0(y, y0, iy);
    upd_img0(z, z0, iz);
}

char *fline(char *f) {
    int n;
    char *r;
    r = fgets(f, BUFSIZ - 1, stdin);
    if (r != NULL) {
        n = strlen(f);
        if (n > 0) f[n - 1] = '\0';
    }
    return r;
}

int main(int argc, const char** argv) {
    int i;
    char f[BUFSIZ]; /* intput file */
    const char *o; /* output dir */
    int Fst;
    float x0, y0, z0;
    float  x,  y,  z;
    int ix, iy, iz; /* periodic image index */
    i = 1;
    X = atof(argv[i++]);
    Y = atof(argv[i++]);
    Z = atof(argv[i++]);
    o = argv[i++];
    
    ix = iy = iz = 0;
    Fst = 1;
    while (fline(/**/ f) != NULL) {
        if (Fst) {
            Fst = 0;
            read_fst(f);
            compute_cm(/**/ &x, &y, &z);
        } else {
            read_rst(f);
            compute_cm(/**/  &x,  &y,  &z);
            upd_img(x, y, z, x0, y0, z0, /*io*/
                    &ix, &iy, &iz);
            shift(ix, iy, iz);
        }
        write_file(o);
        x0 = x; y0 = y; z0 = z;
    }
    if (Fst) fprintf(stderr, "ply2pbc0: warning: no input files\n");
}
