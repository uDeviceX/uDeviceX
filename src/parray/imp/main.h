
void parray_push(const Particle *pp, PaArray *a) {
    a->pp = (const float*)pp;
};

void parray_push_color(const Particle *pp, const int *cc, PaCArray *a) {
    a->pp = (const float*)pp;
    c->cc = cc;
}
