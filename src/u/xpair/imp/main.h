typedef PairFo Fo;
typedef PairPa Pa;

namespace dev {
template <typename Param>
__global__ void main(Param par, Pa a, Pa b, float rnd) {
    Fo f;
    pair_force(par, a, b, rnd, /**/ &f);
    printf("%g %g %g\n", f.x, f.y, f.z);
}
} /* namespace */

void pair(PairParams *par, Pa a, Pa b, int ka, int kb, float rnd) {
    int k0 = ka < kb ? ka : kb;
    int k1 = ka < kb ? kb : ka;

    
    if (k0 == SOLVENT_KIND) {
        PairDPDC pv;
        pair_get_view_dpd_color(par, &pv);
        KL(dev::main, (1, 1), (pv, a, b, rnd));
    }
    else {
        PairDPDLJ pv;
        pair_get_view_dpd_lj(par, &pv);
        if (k0 == SOLID_KIND && k1 == SOLID_KIND)
            pv.ljs *= 2;
        KL(dev::main, (1, 1), (pv, a, b, rnd));
    }
    dSync();
}

void write_pa(Pa a, int kind) {
    fprintf(stderr, "[ %.2g %.2g %.2g ] [ %.2g %.2g %.2g ] [kc: %d %d]\n",
            a.x, a.y, a.z, a.vx, a.vy, a.vz, kind, a.color);
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

void read_pa0(const char *s, Pa *a, int *k) {
    char kind[BUFSIZ], color[BUFSIZ];
    sscanf(s,
           "%f %f %f   %f %f %f   %s %s",
           &a->x, &a->y, &a->z, &a->vx, &a->vy, &a->vz,
           kind, color);
    *k  = decode_kind(kind);
    a->color = decode_color(color);
}

enum {OK, END, FAIL};
int read_pa(Pa *a, int *k) {
    char s[BUFSIZ];
    if (fgets(s, BUFSIZ - 1, stdin) == NULL) return END;
    read_pa0(s, /**/ a, k);
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

static void set_params(PairParams *p) {
    float a[] = {adpd_b, adpd_br, adpd_r};
    float g[] = {gdpd_b, gdpd_br, gdpd_r};
    float s[3];
    for (int i = 0; i < 3; ++i)
        s[i] = sqrt(2*kBT*g[i]*dt);

    pair_set_lj(ljsigma, ljepsilon, p);
    pair_set_dpd(2, a, g, s, p);
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    Pa a, b;
    int ka, kb;
    float rnd;
    PairParams *par;

    pair_ini(&par);    
    set_params(par);
    
    read_rnd(&rnd);
    for (;;) {
        if (read_pa(&a, &ka) == END) break;
        if (read_pa(&b, &kb) == END) break;
        // write_pa(a, ka);
        // write_pa(b, kb);
        pair(par, a, b, ka, kb, rnd);
    }
    pair_fin(par);
    m::fin();
}
