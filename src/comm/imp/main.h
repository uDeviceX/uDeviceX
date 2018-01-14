int post_recv(hBags *b, Comm *com) {
    int i, c, tag;
    for (i = 0; i < NFRAGS; ++i) {
        c = b->capacity[i] * b->bsize;
        tag = com->tags[i];
        MC(m::Irecv(b->data[i], c, MPI_BYTE, com->ranks[i], tag, com->cart, com->rreq + i));
    }
    return 0;
}

static void fail_over(int i, long c, long cap) {
    enum {X, Y, Z};
    int d[3];
    d[X] = frag_i2dx(i); d[Y] = frag_i2dy(i); d[Z] = frag_i2dz(i);
    ERR("over capacity, fragment %d = [%d %d %d]: %ld/%ld",
        i, d[X], d[Y], d[Z], c, cap);
}
int post_send(const hBags *b, Comm *com) {
    int i, n, c, cap, tag;
    for (i = 0; i < NFRAGS; ++i) {
        c = b->counts[i];
        cap = b->capacity[i];
        n = c * b->bsize;
        tag = i;
        if (c > cap) UC(fail_over(i, c, cap));
        MC(m::Isend(b->data[i], n, MPI_BYTE, com->ranks[i], tag, com->cart, com->sreq + i));
    }
    return 0;
}

static void get_counts_bytes(const MPI_Status ss[NFRAGS], /**/ int counts[NFRAGS]) {
    for (int i = 0; i < NFRAGS; ++i)
        MC(m::Get_count(ss + i, MPI_BYTE, counts + i));
}

static void get_counts(const MPI_Status ss[NFRAGS], /**/ hBags *b) {
    int cap, i, c, cc[NFRAGS];
    get_counts_bytes(ss, /**/ cc);

    for (i = 0; i < NFRAGS; ++i) {
        c = cc[i] / b->bsize;
        b->counts[i] = c;
        cap = b->capacity[i];
        if (c > cap) UC(fail_over(i, c, cap));
    }
}

int wait_recv(Comm *com, /**/ hBags *b) {
    MPI_Status ss[NFRAGS];
    MC(m::Waitall(NFRAGS, com->rreq, /**/ ss));
    get_counts(ss, /**/ b);
    return 0;
}

int wait_send(Comm *com) {
    MPI_Status ss[NFRAGS];
    MC(m::Waitall(NFRAGS, com->sreq, ss));
    return 0;
}
