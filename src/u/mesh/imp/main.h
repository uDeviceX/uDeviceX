/* max number of vertices and triangles */
#define NV 1000
#define NT 1000
void add_vert(float x, float y, float z, /**/ float *v) {
    enum {X, Y, Z};
    v[X] = x; v[Y] = y; v[Z] = z;
}
void add_tri (int x, int y, int z, /**/ int4 *t) { (*t).x = x; (*t).y = y; (*t).z = z; }
void piramid(/**/ float *v, int4 *t, int *nt) {
    int i;
    i = 0;
    add_vert(0, 0, 0, /**/ &v[3*i++]);
    add_vert(1, 0, 0, /**/ &v[3*i++]);
    add_vert(0, 1, 0, /**/ &v[3*i++]);
    add_vert(0, 0, 1, /**/ &v[3*i++]);

    i = 0;
    add_tri(0, 1, 2, /**/ &t[i++]);
    add_tri(0, 3, 1, /**/ &t[i++]);
    add_tri(0, 2, 3, /**/ &t[i++]);
    add_tri(1, 3, 2, /**/ &t[i++]);
    *nt = i;
}

void read_point0(const char *s, float *r) {
    enum {X, Y, Z};
    sscanf(s, "%f %f %f", &r[X], &r[Y], &r[Z]);
}

enum {OK, END, FAIL};
int read_point(float *r) {
    char s[BUFSIZ];
    if (fgets(s, BUFSIZ - 1, stdin) == NULL) return END;
    read_point0(s, /**/ r);
    return OK;
}
void write_point(float *r, int inside) {
    enum {X, Y, Z};
    printf("%g %g %g %d\n", r[X], r[Y], r[Z], inside);
}

void main0() {
    float r[3], vv[3*NT];
    int4  tt[NT];
    int nt, inside;
    piramid(vv, tt, &nt);
    while (read_point(r) != END) {
        inside = collision::inside_1p(r, vv, tt, nt);
        write_point(r, inside);
    }
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main0();
    m::fin();
}
