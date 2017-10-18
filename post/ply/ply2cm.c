/*
  Write x y z from ply file to stdio

  Usage:
  ./ply2punto <in.ply>

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NVMAX 8000000
#define NVAR  6 /* x, y, z, vx, vy, vz */
float  buf[NVAR*NVMAX];
FILE* fd;

char line[1024]; /* a line from a file */
int n; /* number of vertices */
int nv = 498;

int comment_line() { /* returns true for a comment line in ply */
    char *pre = "comment";
    return strncmp(pre, line, strlen(pre)) == 0;
}

#define nl() fgets(line, sizeof(line), fd) /* [n]ext [l]ine */
void read_header() {
    nl(); /* ply */
    nl(); /* format binary_little_endian 1.0 */
    do nl(); while (comment_line());
    /* element vertex %nv% */
    sscanf(line, "element vertex %d\n", &n);
    nl(); nl(); nl(); nl(); nl(); nl(); /* property float [xyzuvw] */

    nl(); /* element faces */
    nl(); /* property list int int vertex_index */
    nl(); /* end_header */
}
#undef nl

void read() {
    fread(buf, NVAR*n, sizeof(float), fd);
}

void write() {
    int iv, ib, j;
    float x, y, z;
    float xc, yc, zc;
    for (iv = ib = 0; iv < n; /**/ ) {
        if (iv >= n) break;
        xc = yc = zc = 0;
        for (j = 0; j < nv; j++) {
            x = buf[ib++]; y = buf[ib++]; z = buf[ib++]; ib++; ib++; ib++; iv++;
            xc += x; yc += y; zc += z;
        }
        printf("%g %g %g\n", xc/nv, yc/nv, zc/nv);
    }
}

void read_file(const char* fn) {
    fprintf(stderr, "(ply2punto) reading: %s\n", fn);
    fd = fopen(fn, "r");
    read_header();
    read();
    write();
    fclose(fd);
}

int main(int argc, const char** argv) {
    if (argc != 2) {
        fprintf(stderr, "(ply2punto) Usage: ply2punto <in.ply>\n");
        exit(1);
    }
    read_file(argv[1]);
}
