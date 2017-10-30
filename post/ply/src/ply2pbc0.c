/*

Usage:
nb=498 u.ply2pbc0 [output dir] [ply file]..

 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "endian.h"
#include "rbc_utils.h"

float X, Y, Z;

float xp1, xp2, yp1, yp2, zp1, zp2;

int   nv; /* number of vertices */
int   nf; /* number of faces */

#define NVAR  6 /* x, y, z, vx, vy, vz */
#define NV_PER_FACE 3

float *buf;
float  *xx,  *yy,  *zz;
float *vvx, *vvy, *vvz;
int   *bbv; /* banned vertices */
int   *idx; /* old to new vertices index convertion */
int   *rrbc; /* vertices to RBC index */

int  *bbr; /* banned RBCs */

/* vertex indexes in one face */
int *ff1, *ff2, *ff3;
int *bbf; /* banned faces */

int *ibuf;

FILE* fd_nl;
char line[1024];
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

    bbv = ealloc(nv*sizeof(bbv[0]));
    idx = ealloc(nv*sizeof(idx[0]));
    rrbc = ealloc(nv*sizeof(rrbc[0]));
    bbr =  ealloc(nv*sizeof(bbr[0]));

    ff1 = ealloc(nf*sizeof(ff1[0]));
    ff2 = ealloc(nf*sizeof(ff2[0]));
    ff3 = ealloc(nf*sizeof(ff3[0]));

    bbf = ealloc(nf*sizeof(bbf[0]));
    ibuf = ealloc(nf*(NV_PER_FACE + 1)*sizeof(ibuf[0]));
}

void read_file(const char* fn) {
  fprintf(stderr, "(box) reading: %s\n", fn);
  FILE* fd = safe_fopen(fn, "r");
  read_header(fd);
  balloc();
  read_vertices(fd);
  read_faces(fd);
  fclose(fd);
}

int cnv() { /* get number of not banned vertices */
  int ans, iv;
  for (iv = 0, ans = 0; iv < nv; ++iv)
    if (!bbv[iv]) ans ++;
  return ans;
}

int cnf() { /* get number of not banned faces */
  int ans, ifa;
  for (ifa = 0, ans = 0; ifa < nf; ++ifa)
    if (!bbf[ifa]) ans ++;
  return ans;
}

int reidx(int f) {return idx[f];}
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
  pr("POINTS %d %s\n", cnv(), dataType);

  int ib, iv;
  for (iv=0, ib=0; iv<nv; ++iv) {
    if (bbv[iv]) continue;
    SF(xx[iv]); SF(yy[iv]); SF(zz[iv]);
  }
  fwrite(buf, ib, sizeof(float), fd);
}

#define SI(f) ibuf[ib++] = LongSwap(f);
void write_cells(FILE* fd) {
  int ncell = cnf();
  int size = (1 + NV_PER_FACE)*ncell;
  pr("\n");
  pr("POLYGONS %d %d\n", ncell, size);
  int ib = 0;
  for (int f1, f2, f3, ifa = 0; ifa < nf; ++ifa) {
    if (bbf[ifa]) continue;
    f1 = reidx(ff1[ifa]); f2 = reidx(ff2[ifa]); f3 = reidx(ff3[ifa]);
    SI(NV_PER_FACE); SI(f1); SI(f2); SI(f3);
  }
  fwrite(ibuf, ib, sizeof(int), fd);
}

void write_cells_attributes(FILE* fd) {
  int ncell = cnf();
  const char* const dataType = "float";
  const char* const dataName = "s";

  pr("\n");
  pr("CELL_DATA %d\n", ncell);
  pr("SCALARS %s %s\n", dataName, dataType);
  pr("LOOKUP_TABLE default\n");
  int ifa, f1, irbc, ib = 0, crbc = -1, iv_rbc = 0;
  for (ifa = 0; ifa < nf; ++ifa) { /* color faces with face id inside
                                      one cell */
    if (bbf[ifa]) continue;
    f1 = ff1[ifa]; irbc = rrbc[f1];
    if (irbc != crbc) {iv_rbc = 0; crbc = irbc;} /* new cell */
    else               iv_rbc ++;                /* old cell */
    SF(iv_rbc);
  }
  fwrite(buf, ib, sizeof(float), fd);
}

void write_file(const char* fn) {
  fprintf(stderr, "(rw) writing: %s\n", fn);
  FILE* fd = safe_fopen(fn, "w");
  write_file_version(fd);
  write_header(fd);
  write_format(fd);
  write_vertices(fd);
  write_cells(fd);
  write_cells_attributes(fd);
  fclose(fd);
}

void index_vertices() {
  int iv, ivn;
  for (iv = ivn = 0; iv < nv; ++iv) {
    if (bbv[iv]) continue;
    idx[iv] = ivn++;
  }
}

int main(int argc, const char** argv) {
  int i = 1;
  X = atof(argv[i++]);
  Y = atof(argv[i++]);
  Z = atof(argv[i++]);
  fprintf(stderr, "XYZ: %g %g %g\n", X, Y, Z);

  read_file(argv[1]);
  write_file(argv[2]);
}
