void mesh_area_ini(MeshRead *mesh, MeshArea **pq) {
    int nt;
    MeshArea *q;
    EMALLOC(1, &q);
    nt = mesh_read_get_nt(mesh);
    EMALLOC(  nt, &q->tt);

    q->nt = nt;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);

    *pq = q;
}

void mesh_area_fin(MeshArea *q) {
    EFREE(q->tt);
    EFREE(q);
}

static double area0(double a[3], double b[3], double c[3]) {
    enum {X, Y, Z};
    return  (+a[X]*(b[Y]*c[Z]-b[Z]*c[Y])
             -a[Y]*(b[X]*c[Z]-b[Z]*c[X])
             +a[Z]*(b[X]*c[Y]-b[Y]*c[X]));
}
static void get(Positions *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(positions_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double area(int nt, int4 *tt, Positions *p, int offset) {
    enum {X, Y, Z};
    int i, ia, ib, ic;
    double a[3], b[3], c[3], sum;
    KahanSum *kahan_sum;
    kahan_sum_ini(&kahan_sum);
    for (i = 0; i < nt; i++) {
        ia = tt[i].x; ib = tt[i].y; ic = tt[i].z;
        UC(get(p, ia + offset, /**/ a));
        UC(get(p, ib + offset, /**/ b));
        UC(get(p, ic + offset, /**/ c));
        kahan_sum_add(kahan_sum, area0(a, b, c));
    }
    sum = kahan_sum_get(kahan_sum);
    return sum/6;
}
double mesh_area_apply0(MeshArea *q, Positions *p) {
    int nt, offset;
    int4 *tt;
    nt = q->nt; tt = q->tt; offset = 0;
    return area(nt, tt, p, offset);
}
