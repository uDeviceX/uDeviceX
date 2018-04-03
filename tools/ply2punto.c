#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void usg() {
    fprintf(stderr, "ply2punto 1.ply 2.ply .. > punto.dat\n");
    exit(0);
}

#define NVMAX 1000000
#define NVAR  6 /* x, y, z, vx, vy, vz */
float  buf[NVAR*NVMAX];
FILE* fd;

char line[FILENAME_MAX]; /* a line from a file */
int nv; /* number of vertices */

int comment_line() {
    const char pre[] = "comment";
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
    int iv, ib;
    float x, y, z;
    fread(buf, NVAR*nv, sizeof(float), fd);
    for (iv = ib = 0; iv < nv; ++iv) {
        x = buf[ib++], y = buf[ib++], z = buf[ib++];
        printf("%g %g %g\n", x, y, z);
        ib++; ib++; ib++; /* skip vx, vy, vz */
    }
}

FILE* efopen(const char *p, const char *m) {
    FILE *f;
    f = fopen(p, m);
    if (f == NULL) {
        fprintf(stderr, "ply2punto: fail to open %s\n", p);
        exit(2);
    }
    return f;
}

void read_file(const char* fn) {
    fd = efopen(fn, "r");
    read_header();
    read_write_vertices();
    fclose(fd);
}

int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

void help(int c, const char** a) {
    if (c > 1 && eq(a[1], "-h")) usg();
}

int main(int argc, const char** argv) {
    int i;
    help(argc, argv);
    for (i = 1; i < argc; i++) {
        if (i > 1) printf("\n");
        read_file(argv[i]);
    }
    return 0;
}
