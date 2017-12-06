/*

ply to vtk converter

# TEST: ply2vtk.t1
# nb=498 ./ply2vtk test_data/rbc.org.ply     rbc.out.vtk
#
# TEST: ply2vtk.t2
# nb=498 ./ply2vtk test_data/rbc.comment.ply rbc.out.vkt

*/

#include <math.h>
#include "endian.h"
#include "rbc_utils.h"

enum {X, Y, Z};

int nb; /* number of vertices in one RBC (number of beads) */
int nv; /* number of vertices */
int nf; /* number of faces */

#define NVMAX 10000000
#define NFMAX 10000000

#define NVAR  6 /* x, y, z, vx, vy, vz */
#define NV_PER_FACE 3

float buf[NVAR*NVMAX]; /* buffer for vertices */
float  xx[NVMAX],  yy[NVMAX],  zz[NVMAX];
float  vx[NVMAX],  vy[NVMAX],  vz[NVMAX];
int   rrbc[NVMAX]; /* vertices to RBC index */

/* vertex indexes in one face */
int ff1[NFMAX], ff2[NFMAX], ff3[NFMAX];

int ibuf[NFMAX*(NV_PER_FACE + 1)]; /* buffer for faces */

void init() {
  nb  = env2d("nb");
}

FILE* fd_nl;
void init_nl(FILE* fd) {fd_nl = fd;}
char line[1024];
void nl() { /* read [n]ext [l]ine; sets `line' */
  fgets(line, sizeof(line), fd_nl);
}

bool comment_line() { /* returns true for a comment line in ply */
  auto pre = "comment";
  return strncmp(pre, line, strlen(pre)) == 0;
}

void read_header(FILE* fd) {
  init_nl(fd);
  nl(); /* ply */
  nl(); /* format binary_little_endian 1.0 */
  do nl(); while (comment_line());
  /* element vertex %nv% */
  sscanf(line, "element vertex %d\n", &nv);
  nl(); nl(); nl(); nl(); nl(); nl(); /* property float [xyzuvw] */

  nl(); sscanf(line, "element face %d\n", &nf);
  nl(); /* property list int int vertex_index */
  nl(); /* end_header */
}

void read_vertices(FILE* fd) {
  fread(buf, NVAR*nv, sizeof(float), fd);
  for (int iv=0, ib=0; iv<nv; ++iv) {
     xx[iv] = buf[ib++];  yy[iv] = buf[ib++];  zz[iv] = buf[ib++];
     vx[iv] = buf[ib++];  vy[iv] = buf[ib++]; vz[iv] = buf[ib++];
  }
}

void read_faces(FILE* fd) {
  fread(ibuf, nf*(NV_PER_FACE+1), sizeof(int), fd);
  int nvpf;
  for (int ifa = 0, ib = 0; ifa < nf; ++ifa) {
    nvpf = ibuf[ib++];
    ff1[ifa] = ibuf[ib++]; ff2[ifa] = ibuf[ib++]; ff3[ifa] = ibuf[ib++];
  }
}

void read_file(const char* fn) {
  FILE* fd = safe_fopen(fn, "r");
  read_header(fd);
  read_vertices(fd);
  read_faces(fd);
  fclose(fd);
}

#define pr(...) fprintf (fd, __VA_ARGS__)
void write_file_version(FILE* fd) {
  pr("# vtk DataFile Version 2.0\n");
}

void write_header(FILE* fd) {
    pr("Created with vrbc utils\n");
};

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
};

#define SI(f) ibuf[ib++] = LongSwap(f);
void write_cells(FILE* fd) {
  int ncell = nf;
  int size = (1 + NV_PER_FACE)*ncell;
  pr("\n");
  pr("POLYGONS %d %d\n", ncell, size);
  int ib = 0;
  for (int f1, f2, f3, ifa = 0; ifa < nf; ++ifa) {
    f1 = ff1[ifa]; f2 = ff2[ifa]; f3 = ff3[ifa];
    SI(NV_PER_FACE); SI(f1); SI(f2); SI(f3);
  }
  fwrite(ibuf, ib, sizeof(int), fd);
};

void write_cells_header(FILE *fd) {
  int ncell = nf;
  pr("\n");
  pr("CELL_DATA %d\n", ncell);
}

void write_cells_id(FILE* fd) {
  const char* const dataType = "float";
  const char* const dataName = "s";
  pr("SCALARS %s %s\n", dataName, dataType);
  pr("LOOKUP_TABLE default\n");
  int ifa, f1, irbc, ib = 0, crbc = -1, iv_rbc = 0;
  for (ifa = 0; ifa < nf; ++ifa) { /* color faces with face id inside
				      one cell */
    f1 = ff1[ifa]; irbc = rrbc[f1];
    if (irbc != crbc) {iv_rbc = 0; crbc = irbc;} /* new cell */
    else               iv_rbc ++;                /* old cell */
    SF(iv_rbc);
  }
  fwrite(buf, ib, sizeof(float), fd);
}

void write_cells_velocity(FILE *fd, float *vv, char* name) { /* D: dimension */
  int ncell = nf;
  const char* const dataType = "float";
  pr("\n");
  pr("SCALARS %s %s\n", name, dataType);
  pr("LOOKUP_TABLE default\n");
  float v;
  int ifa, f1, f2, f3, ib = 0;
  for (ifa = 0; ifa < nf; ++ifa) {
    f1 = ff1[ifa]; f2 = ff2[ifa]; f3 = ff3[ifa];
    v = (vv[f1] + vv[f2] + vv[f3]) / 3;
    SF(v);
  }
  fwrite(buf, ib, sizeof(float), fd);
}

void write_file(const char* fn) {
  FILE* fd = safe_fopen(fn, "w");
  write_file_version(fd);
  write_header(fd);
  write_format(fd);
  write_vertices(fd);
  write_cells(fd);
  write_cells_header(fd);
  write_cells_id(fd);
  write_cells_velocity(fd, vx, "vx");
  write_cells_velocity(fd, vy, "vy");
  write_cells_velocity(fd, vz, "vz");  
  fclose(fd);
}

void build_rbc_index() {
  int iv, rbc, crbc;
  for (iv = rbc = crbc = 0; iv < nv; ++iv, ++crbc) {
    if (crbc == nb) {rbc++; crbc = 0;} /* new RBC */
    rrbc[iv] = rbc;
  }
}

int main(int argc, const char** argv) {
  init();

  read_file(argv[1]);
  build_rbc_index();

  write_file(argv[2]);
}
