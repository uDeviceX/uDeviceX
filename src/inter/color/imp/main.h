void inter_color_ini(GenColor **c) {
    UC(emalloc(sizeof(GenColor), (void**) c));
}

void inter_color_fin(GenColor *c) {
    UC(efree(c));
}

void inter_color_set_drop(float R, GenColor *c) {
    c->kind = DROP;
    c->R = R;
}

void inter_color_set_uniform(GenColor*) {
    c->kind = UNIF;
}
