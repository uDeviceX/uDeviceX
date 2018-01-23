void sdf_ini(Sdf **pq) {
    Sdf *q;
    UC(emalloc(sizeof(Sdf), (void**)&q));
    UC(array3d_ini(&q->arr, XTE, YTE, ZTE));
    UC(  tform_ini(&q->t));
    *pq = q;
}

void sdf_fin(Sdf *q) {
    UC(array3d_fin(q->arr));
    UC(  tex3d_fin(q->tex));
    UC(  tform_fin(q->t));
    UC(efree(q));
}

void sdf_bounce(const Wvel_v *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp) {
    UC(bounce_back(wv, c, sdf, n, /**/ pp));
}


void sdf_to_view(const Sdf *q, /**/ Sdf_v *v) {
    tex3d_to_view(q->tex, &v->tex);
    tform_to_view(q->t  , &v->t);
}
