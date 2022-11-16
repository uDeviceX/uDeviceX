void mesh_cylindrical_ini(int nv, MeshCylindrical **pq) {
    MeshCylindrical *q;
    EMALLOC(1, &q);
    EMALLOC(3*nv, &q->rr);
    q->nv = nv;
    *pq = q;
}

void mesh_cylindrical_fin(MeshCylindrical *q) {
    EFREE(q->rr);
    EFREE(q);
}

static void xyz2cyl(double x, double y, double z, /**/ double *pr, double *pphi, double *pz) {
    double r, phi;
    r = sqrt(x*x + y*y);
    phi = atan2(y, x);
    *pr = r; *pphi  = phi; *pz = z;
}
static void apply(int n, double *rr, /**/ double *r, double *phi, double *z) {
    enum {X, Y, Z};
    int i;
    for (i = 0; i < n; i++) {
        xyz2cyl(rr[X], rr[Y], rr[Z], /**/ r, phi, z);
        rr += 3; r++; phi++; z++;
    }
}
void mesh_cylindrical_apply(MeshCylindrical *q, int m, Vectors *pos, /**/ double *r, double *phi, double *z) {
    int i, nv, offset;
    double *rr;
    nv = q->nv; rr = q->rr; offset = 0;    
    for (i = 0; i < m; i++) {
        UC(to_com(nv, offset, pos, /**/ rr));
        apply(nv, rr, /**/ r, phi, z);
        rr += 3*nv; r += nv; phi += nv; z += nv; offset += nv;
    }    
}
