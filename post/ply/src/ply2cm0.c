#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NVMAX 20000000
#define NVAR  6 /* x, y, z, vx, vy, vz */

void usg() {
    fprintf(stderr, "usage: ply2punto <in.ply>\n");
    exit(2);
}

static float  *buf;
static FILE* fd;

static char line[BUFSIZ]; /* a line from a file */
static int n; /* number of vertices */
static int nv = 498;

int comment_line() { /* returns true for a comment line in ply */
    char *pre = "comment";
    return strncmp(pre, line, strlen(pre)) == 0;
}

void efgets(char *s, int size, FILE *f) {
    if (NULL != fgets(s, size, f)) return;
    fprintf(stderr, "fail to read string\n");
    exit(2);
}

#define nl() efgets(line, sizeof(line), fd) /* [n]ext [l]ine */
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

void eread(void *p, size_t size, size_t nmemb, FILE *f) {
    if (fread(p, size, nmemb, f)) return;
    fprintf(stderr, "fail to read binary data\n");
    exit(2);
}
void read() { eread(buf, NVAR*n, sizeof(float), fd); }
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

FILE *efopen(const char *path, const char *mode) {
    FILE *f;
    f = fopen(path, mode);
    if (f == NULL) {
        fprintf(stderr, "fail to open file %s\n", path);
        exit(-2);
    }
    return f;
}

void balloc() {
    size_t sz;
    sz = NVAR * n * sizeof(buf[0]);
    buf = malloc(sz);
    if (buf == NULL) {
        fprintf(stderr, "fail to alloc: %ld\n", sz);
        exit(2);
    }
    
}

void bfree() { free(buf); }


void read_file(const char* fn) {
    fd = efopen(fn, "r");
    read_header();
    balloc();
    read();
    fclose(fd);
    write();
    bfree();
}

int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }
int main(int argc, const char** argv) {
    if (argc > 1 && eq(argv[1], "-h")) usg();
    if (argc != 2) usg();
    
    read_file(argv[1]);
}
