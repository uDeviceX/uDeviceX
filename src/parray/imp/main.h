void parray_push_pp(const Particle *pp, PaArray *a) {
    a->pp = (const float*)pp;
    a->colors = false;
};

void parray_push_cc(const int *cc, PaArray *a) {
    a->cc = cc;
    a->colors = true;
}

bool parray_is_colored(const PaArray *a) {
    return a->colors;
}

void parray_get_view(const PaArray *a, PaArray_v *v) {
    v->pp = a->pp;
}

void parray_get_view(const PaArray *a, PaCArray_v *v) {
    v->pp = a->pp;
    v->cc = a->cc;
}
