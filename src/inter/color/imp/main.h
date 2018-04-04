void inter_color_ini(GenColor **c) {
    GenColor *gc;
    EMALLOC(1, c);
    gc = *c;
    gc->kind = NONE;
}

void inter_color_fin(GenColor *c) {
    EFREE(c);
}

void inter_color_set_drop(float R, GenColor *c) {
    c->kind = DROP;
    c->R = R;
}

void inter_color_set_uniform(GenColor *c) {
    c->kind = UNIF;
}

void inter_color_apply_hst(const Coords *coords, const GenColor *gc, int n, const Particle *pp, /**/ int *cc) {
    switch (gc->kind) {
    case UNIF:
        set_color_unif(n, /**/ cc);
        break;
    case DROP:
        set_color_drop(gc->R, coords, n, pp, /**/ cc);
        break;
    case NONE:
        break;
    default:
        ERR("Unrecognised kind <%d>", gc->kind);
        break;
    };
}

void inter_color_apply_dev(const Coords *coords, const GenColor *gc, int n, const Particle *pp, /**/ int *cc) {
    int *cc_hst;
    Particle *pp_hst;
    size_t szc, szp;

    if (gc->kind == NONE) return;
    
    szc = n * sizeof(int);
    szp = n * sizeof(Particle);

    EMALLOC(n, &cc_hst);
    EMALLOC(n, &pp_hst);

    CC(d::Memcpy(pp_hst, pp, szp, D2H));

    UC(inter_color_apply_hst(coords, gc, n, pp_hst, /**/ cc_hst));

    CC(d::Memcpy(cc, cc_hst, szc, H2D));
    
    EFREE(pp_hst);
    EFREE(cc_hst);
}
