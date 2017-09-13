void post_recv(hBags *b, Stamp *s) {
    int i, c, tag;
    for (i = 0; i < NFRAGS; ++i) {
        c = b->capacity[i] * b->bsize;
        tag = s->bt + s->tags[i];
        MC(m::Irecv(b->data[i], c, MPI_BYTE, s->ranks[i], tag, s->cart, s->rreq + i));
    }
}

void post_send(hBags *b, Stamp *s) {
    int i, n, tag;
    for (i = 0; i < NFRAGS; ++i) {
        n = b->counts[i] * b->bsize;
        tag = s->bt + i;
        MC(m::Isend(b->data[i], n, MPI_BYTE, s->ranks[i], tag, s->cart, s->sreq + i));
    }
}

static void get_counts_bytes(const MPI_Status ss[NFRAGS], /**/ int counts[NFRAGS]) {
    for (int i = 0; i < NFRAGS; ++i)
        MC(m::Get_count(ss + i, MPI_BYTE, counts + i));
}

static void get_counts(const MPI_Status ss[NFRAGS], /**/ hBags *b) {
    int i, c, cc[NFRAGS];
    get_counts_bytes(ss, /**/ cc);

    for (i = 0; i < NFRAGS; ++i) {
        c = cc[i] / b->bsize;
        b->counts[i] = c;
        if (c >= b->capacity[i])
            ERR("recv more than capacity.");
    }
}

void wait_recv(Stamp *s, /**/ hBags *b) {
    MPI_Status ss[NFRAGS];
    MC(m::Waitall(NFRAGS, s->rreq, /**/ ss));
    get_counts(ss, /**/ b);
}

void wait_send(Stamp *s) {
    MPI_Status ss[NFRAGS];
    MC(m::Waitall(NFRAGS, s->sreq, ss));
}
