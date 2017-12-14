typedef forces::Fo Fo;
typedef forces::Pa Pa;

namespace dev {
__global__ void main(Pa a, Pa b, float rnd) {
    Fo f;
    forces::gen(a, b, rnd, /**/ &f);
    printf("%g %g %g\n", f.x, f.y, f.z);
}
} /* namespace */

void pair(Pa a, Pa b, float rnd) {
    KL(dev::main, (1, 1), (a, b, rnd));
    dSync();
}

void write_pa(Pa a) {
    fprintf(stderr, "[ %.2g %.2g %.2g ] [ %.2g %.2g %.2g ] [kc: %d %d]\n",
            a.x, a.y, a.z, a.vx, a.vy, a.vz, a.kind, a.color);
}

int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }
void err(const char *s) {
    fprintf(stderr, "%s\n", s);
    exit(2);
}
int decode_kind(const char *s) {
    int r;
    if      (eq(s, "SOLVENT") || eq(s, "O") || eq(s, "0")) r = SOLVENT_KIND;
    else if (eq(s, "SOLID")   || eq(s, "S") || eq(s, "1")) r = SOLID_KIND;
    else if (eq(s, "WALL")    || eq(s, "W") || eq(s, "2")) r = WALL_KIND;
    else err("unknow kind");
    return r;
}

int decode_color(char *s) {
    int r;
    if      (eq(s, "BLUE") || eq(s, "B") || eq(s, "0")) r = BLUE_COLOR;
    else if (eq(s, "RED")  || eq(s, "R") || eq(s, "1")) r = RED_COLOR;    
    else err("unknow color");
    return r;
}

void read_pa0(const char *s, Pa *a) {
    char kind[BUFSIZ], color[BUFSIZ];
    sscanf(s,
           "%f %f %f   %f %f %f   %s %s",
           &a->x, &a->y, &a->z, &a->vx, &a->vy, &a->vz,
           kind, color);
    a->kind  = decode_kind(kind);
    a->color = decode_color(color);
}

enum {OK, END, FAIL};
int read_pa(Pa *a) {
    char s[BUFSIZ];
    if (fgets(s, BUFSIZ - 1, stdin) == NULL) return END;
    read_pa0(s, /**/ a);
    return OK;
}

void read_rnd(/**/ float *prnd) {
    char *s;
    float rnd;

    s = getenv("RND");
    if (s == NULL) rnd = 0;
    else           rnd = atof(s);
    *prnd = rnd;
}

void main0() {
    Pa a, b;
    float rnd;
    read_rnd(&rnd);
    for (;;) {
        if (read_pa(&a) == END) break;
        if (read_pa(&b) == END) break;
        pair(a, b, rnd);
    }
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    main0();
    m::fin();
}
