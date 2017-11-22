#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void usg() {
    fprintf(stderr, "ply2punto 1.ply 2.ply .. > punto.dat\n");
    exit(0);
}

#define NVT   3 /* vertices per triangle */
#define NVMAX 1000000
#define NVAR  6 /* x, y, z, vx, vy, vz */
FILE* fd;

char line[1024]; /* a line from a file */
int nv, nt; /* number of vertices and triangles */

int commentp() { return strcmp("comment", line) == 0; }
#define nl() fgets(line, sizeof(line), fd) /* [n]ext [l]ine */
void header() {
    nl(); /* ply */
    nl(); /* format binary_little_endian 1.0 */
    do nl(); while (commentp());
    /* element vertex %nv% */
    sscanf(line, "element vertex %d\n", &nv);
    nl(); nl(); nl(); nl(); nl(); nl(); /* property float [xyzuvw] */
    nl(); sscanf(line, "element face %d\n", &nt);
    nl(); /* property list int int vertex_index */
    nl(); /* end_header */
}
#undef nl

void verts0(float *buf) {
    int i, b;
    float x, y, z;
    efread(buf, NVAR*nv, sizeof(float), fd);
    for (i = b = 0; i < nv; ++i) {
        x = buf[b++], y = buf[b++], z = buf[b++];
        printf("%g %g %g\n", x, y, z);
        b++; b++; b++; /* skip vx, vy, vz */
     }
}
void verts() {
    float *buf;
    int sz = nv*NVAR;
    buf = malloc(nv*sizeof(buf));
    verts0(buf);
    free(buf);
}

void efread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t r;
    r = fread(ptr, size, nmemb, stream);
    if (r == 0) {
        fprintf(stderr, "ply2off: fails to read: %ld/%ld\n", size, nmemb);
        exit(2);
    }
}

void tris0(int *buf) {
    int i, b;
    efread(buf, nt*(NVT+1), sizeof(buf[0]), fd);
    for (i = b = 0; i < nt; ++i) {
        b++;
        printf("%d: %d %d %d\n", i, buf[b++], buf[b++], buf[b++]);
    }
}
void tris() {
    int *buf;
    int sz = nt*(NVT+1);
    buf = malloc(nv*sizeof(buf));
    tris0(buf);
    free(buf);
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

void file(const char* fn) {
    fd = efopen(fn, "r");
    header();
    verts();
    tris();
    fclose(fd);
}

int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }
void help(int c, const char** a) {
    if (c > 1 && eq(a[1], "-h")) usg();
}

int main(int argc, const char** argv) {
    int i;
    help(argc, argv);
    file(argv[1]);
    return 0;
}
