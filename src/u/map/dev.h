__global__ void main() {
    using namespace map;
    hforces::Frag frag;
    M m;

    int xs, ys, zs;
    xs = ys = zs = 2;
    int s[] = {0, 3, 6, 9, 12, 15, 18};
    frag.start = s;
    frag.dx = -1; frag.dy = 0; frag.dz = 0;
    int i, j, k, r;
    j = 0;
    ini0(&frag, 1.9, 1.0, 1.0, xs, ys, zs, &m);
    for (k = 0; k < 30; k++) {
        r = nxt0(&m, &i, &j, xs, ys, zs);
        printf("%s\n",
               r == OLD ? "OLD" :
               r == NEW ? "NEW" :
               r == END ? "END" :
               "UNKONW");
        printf("d: %d %d %d   %d\n",
               m.dx, m.dy, m.dz, j);
        if (r == END) break;
    }
    printf("end\n");
}
