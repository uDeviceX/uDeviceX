void inter_color_ini(GenColor **c) {
    GenColor *gc;
    UC(emalloc(sizeof(GenColor), (void**) c));
    gc = *c;
    gc->kind = -1;
}

void inter_color_fin(GenColor *c) {
    UC(efree(c));
}

void inter_color_set_drop(float R, GenColor *c) {
    c->kind = DROP;
    c->R = R;
}

void inter_color_set_uniform(GenColor *c) {
    c->kind = UNIF;
}

void inter_color_apply(const Coords *coords, const GenColor *gc, int n, const Particle *pp, /**/ int *cc) {
    switch (gc->kind) {
    case UNIF:
        set_color_unif(n, /**/ cc);
        break;
    case DROP:
        set_color_drop(gc->R, coords, n, pp, /**/ cc);
        break;
    default:
        ERR("Unrecognised kind <%d>", gc->kind);
        break;
    };
}
