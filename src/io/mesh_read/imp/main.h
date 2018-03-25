struct Q { int4 *d; }; /* queue */
static void q_push(Q *q, int i0, int i1, int i2, int i3) {
    int4 *d;
    d = q->d;
    d->x = i0; d->y = i1; d->z = i2; d->w = i3;
    (q->d)++;
}

static void reg(int i1, int i2, Edg *nxt, Edg *seen, /**/ Q *q) {
    int i0, i3;
    if (e_valid(seen, i1, i2)) return;
    if (e_valid(seen, i2, i1)) return;
    i0 = e_get(nxt, i1, i2);
    i3 = e_get(nxt, i2, i1); /* previ_ous */
    q_push(q, i0, i1, i2, i3);
    UC(e_set(seen, i1, i2, 1));
}
static void ini_dd0(int nt, const int4 *tt, int nv, int md, /**/ int4 *dd) {
    Q q;
    Edg *nxt, *seen;
    int i, i0, i1, i2;
    q.d = dd;

    UC(e_ini(md, nv, &nxt));
    UC(e_ini(md, nv, &seen));

    for (i = 0; i < nt; i++) { /* save who is next for every edge */
        i0 = tt[i].x; i1 = tt[i].y; i2 = tt[i].z;
        e_set(nxt, i0, i1, i2);
        e_set(nxt, i1, i2, i0);
        e_set(nxt, i2, i0, i1);
    }

    for (i = 0; i < nt; i++) { /* register dihidrals */
        i0 = tt[i].x; i1 = tt[i].y; i2 = tt[i].z;
        reg(i0, i1, nxt, /**/ seen, &q);
        reg(i1, i2, nxt, /**/ seen, &q);
        reg(i2, i0, nxt, /**/ seen, &q);
    }

    e_fin(nxt); e_fin(seen);
}

static void ini_dd(/**/ MeshRead *q) {
    int nt, nv, nd, md;
    const int4 *tt;
    nt = mesh_read_get_nt(q);
    nv = mesh_read_get_nv(q);
    nd = mesh_read_get_ne(q);
    md = mesh_read_get_md(q);
    tt = mesh_read_get_tri(q);
    q->nd = nd;
    EMALLOC(nd, &q->dd);
    ini_dd0(nt, tt, nv, md, /**/ q->dd);
}
                   
void mesh_read_ini_off(const char *path, MeshRead **pq) {
    FILE *f;
    MeshRead *q;
    EMALLOC(1, &q);    
    UC(efopen(path, "r", /**/ &f));
    UC(read_off(f, path, /**/ q));
    UC(efclose(f));
    msg_print("read off '%s'", path);
    ini_dd(q);
    *pq = q;
}

void mesh_read_ini_ply(const char *path, MeshRead **pq) {
    FILE *f;
    MeshRead *q;
    EMALLOC(1, &q);    
    UC(efopen(path, "r", /**/ &f));
    UC(read_ply(f, path, /**/ q));
    UC(efclose(f));
    msg_print("read ply '%s'", path);
    ini_dd(q);
    *pq = q;
}

void mesh_read_fin(MeshRead* q) {
    EFREE(q->rr); EFREE(q->tt); EFREE(q->dd); EFREE(q);
}
int mesh_read_get_nv(const MeshRead *q) { return q->nv; }
int mesh_read_get_nt(const MeshRead *q) { return q->nt; }
int mesh_read_get_ne(const MeshRead *q) {
    int nt;
    nt = q->nt;
    if (nt % 2 != 0) ERR("nt=%d % 2 != 0", nt);
    return 3 * nt / 2;
}

static int amax(int *a, int n) {
    int i, m;
    if (n <= 0) ERR("amax called with size: %d\n", n);
    m = a[0];
    for (i = 1; i < n; i++)
        if (a[i] > m) m = a[i];
    return m;
}
int mesh_read_get_md(const MeshRead *q) {
    int *d;
    int i, m, nt, nv;
    int x, y, z;
    int4 *tt;
    nt = q->nt; nv = q->nv; tt = q->tt;

    if (nv == 0) {
        msg_print("mesh_get_md called for nv = 0");
        return 0;
    }
    UC(emalloc(nv*sizeof(d[0]), (void**)&d));
    for (i = 0; i < nv; i++) d[i] = 0;
    for (i = 0; i < nt; i++) {
        x = tt[i].x; y = tt[i].y; z = tt[i].z;
        d[x]++; d[y]++; d[z]++;
    }
    m = amax(d, nv);
    UC(efree(d));
    return m;
}

const float *mesh_read_get_vert(const MeshRead *q) { return q->rr; }
const int4  *mesh_read_get_tri(const MeshRead *q) { return q->tt; }
const int4  *mesh_read_get_dih(const MeshRead *q) { return q->dd; }
