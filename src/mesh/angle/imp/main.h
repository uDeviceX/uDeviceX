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
    i3 = e_get(nxt, i2, i1); /* previous */
    q_push(q, i0, i1, i2, i3);
    UC(e_set(seen, i1, i2, 1));
}
static void ini_dd(int nt, const int4 *tt, int nv, int md, /**/ int4 *dd) {
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

void mesh_angle_ini(MeshRead *mesh, MeshAngle **pq) {
    int nv, nt, nd, md;
    MeshAngle *q;
    UC(nt = mesh_read_get_nt(mesh));
    UC(nv = mesh_read_get_nv(mesh));
    UC(nd = mesh_read_get_ne(mesh));
    UC(md = mesh_read_get_md(mesh));
    EMALLOC(1, &q);
    EMALLOC(nd, &q->dd);
    UC(ini_dd(nt, mesh_read_get_tri(mesh), nv, md, /**/ q->dd));
    q->nd = nd;
    *pq = q;
}

void mesh_angle_fin(MeshAngle *q) {
    EFREE(q->dd); EFREE(q);
}
void mesh_angle_apply(MeshAngle*, int, Positions*, /**/ double*) {
}
