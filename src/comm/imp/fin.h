void fin(/**/ Bags *b) {
    for (int i = 0; i < NBAGS; ++i)
        if (b->hst) CC(d::FreeHost (b->hst));
}

void fin(/**/ Stamp *s) {
    MC(m::Comm_free(&s->cart));
}
