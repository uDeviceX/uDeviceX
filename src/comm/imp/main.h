void post_recv(Bags *b, Stamp *s) {
    for (int i = 0; i < NFRAGS; ++i) {
        size_t c = b->counts[i] * b->bsize; // TODO: capacity
        MC(m::Irecv(b->hst[i], c, MPI_BYTE, s->rnks[i], s->bt + i, s->cart, s->req + i));
    }
}

void post_send(Bags *b, Stamp *s) {
    for (int i = 0; i < NFRAGS; ++i) {
        size_t n = b->counts[i] * b->bsize;
        MC(m::Isend(b->hst[i], n, MPI_BYTE, s->anks[i], s->bt + i, s->cart, s->req + i));
    }
}

void wait_all(Stamp *s) {
    MPI_Status statuses[NFRAGS];
    MC(m::Waitall(NFRAGS, s->req, statuses));
}
