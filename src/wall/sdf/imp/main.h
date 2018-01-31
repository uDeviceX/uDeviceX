static int3 texture_grid_size(int3 L, int3 M) {
    enum {PAD = 16};
    int3 T;
    int x, y, z; /* grid extents in real space */
    int ty, tz;
    
    x = L.x + 2 * M.x;
    y = L.y + 2 * M.y;
    z = L.z + 2 * M.z;
    
    T.x = PAD * PAD;
    ty = ceiln(y * x, T.x);
    tz = ceiln(z * x, T.x);
    T.y = PAD * ceiln(ty, PAD);
    T.z = PAD * ceiln(tz, PAD);

    return T;
};

void sdf_ini(Sdf **pq) {
    Sdf *q;
    UC(emalloc(sizeof(Sdf), (void**)&q));
    UC(array3d_ini(&q->arr, XTE, YTE, ZTE));
    UC(  tform_ini(&q->t));
    q->cheap_threshold = - sqrt(3.f) * ((float)XSIZE_WALLCELLS / (float)XTE);
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
    v->cheap_threshold = q->cheap_threshold;
}
