void mesh_spherical_ini(int nv, MeshSpherical **pq) {
    MeshSpherical *q;
    EMALLOC(1, &q);
    EMALLOC(3*nv, &q->rr);
    q->nv = nv;
    *pq = q;
}

void mesh_spherical_fin(MeshSpherical *q) {
    EFREE(q->rr);
    EFREE(q);
}

static void xyz2sph(double x, double y, double z, /**/ double *pr, double *ptheta, double *pphi) {
    double r, theta, phi;
    
    *pr = r; *ptheta = theta; *pphi = phi;
}
static void apply(int n, double *rr, /**/ double *r, double *theta, double *phi) {
    enum {X, Y, Z};
    int i;
    for (i = 0; i < n; i++) {
        xyz2sph(rr[X], rr[Y], rr[Z], /**/ r, theta, phi);
        rr += 3; r++; theta++; phi++;
    }
}
void mesh_spherical_apply(MeshSpherical *q, int m, Vectors *pos, /**/ double *r, double *theta, double *phi) {
    int i, nv, offset;
    double *rr;
    nv = q->nv; rr = q->rr; offset = 0;    
    for (i = 0; i < m; i++) {
        UC(to_com(nv, offset, pos, /**/ rr));
        apply(nv, rr, /**/ r, theta, phi);
        rr += 3*nv; r += nv; theta += nv; phi += nv; offset += nv;
    }    
}
