/*
  Write x y z from ply file to stdio

  Usage:
  ./ply2punto <in.ply>

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NVMAX 1000000
#define NVAR  6 /* x, y, z, vx, vy, vz */
float  buf[NVAR*NVMAX];
FILE* fd;

char line[1024]; /* a line from a file */
int nv; /* number of vertices */

bool comment_line() { /* returns true for a comment line in ply */
  auto pre = "comment";
  return strncmp(pre, line, strlen(pre)) == 0;
}

#define nl() fgets(line, sizeof(line), fd) /* [n]ext [l]ine */
void read_header() {
  nl(); /* ply */
  nl(); /* format binary_little_endian 1.0 */
  do nl(); while (comment_line());
  /* element vertex %nv% */
  sscanf(line, "element vertex %d\n", &nv);
  nl(); nl(); nl(); nl(); nl(); nl(); /* property float [xyzuvw] */

  nl(); /* element faces */
  nl(); /* property list int int vertex_index */
  nl(); /* end_header */
}
#undef nl

void read_write_vertices() {
  fread(buf, NVAR*nv, sizeof(float), fd);
  for (int iv = 0, ib = 0; iv < nv; ++iv) {
    auto x = buf[ib++], y = buf[ib++], z = buf[ib++];
    printf("%g %g %g\n", x, y, z);
    ib++; ib++; ib++; /* skip vx, vy, vz */
  }
}

void read_file(const char* fn) {
  fprintf(stderr, "(ply2punto) reading: %s\n", fn);
  fd = fopen(fn, "r");
  read_header();
  read_write_vertices();
  fclose(fd);
}

int main(int argc, const char** argv) {
  if (argc != 2) {
    fprintf(stderr, "(ply2punto) Usage: ply2punto <in.ply>\n");
    exit(1);
  }
  read_file(argv[1]);
}
