namespace sub {
void sendF(int ranks[26], x::TicketTags t, int counts[26], Fop26 FF_pi) {
    int i, n;
    reqsendA.resize(26);
    for (i = 0; i < 26; ++i) {
        n = counts[i];
        MC(m::Isend(FF_pi.d[i], 3 * n, MPI_FLOAT, ranks[i], t.btf + i, m::cart, &reqsendA[i]));
    }
}

void sendC(int dranks[26], x::TicketTags t, int counts[26]) {
    int i;
    reqsendC.resize(26);
    for (i = 0; i < 26; ++i)
        MC(m::Isend(counts + i, 1, MPI_INTEGER, dranks[i], t.btc + i, m::cart, &reqsendC[i]));
}

void sendP(int ranks[26], x::TicketTags t, x::TicketPinned ti, Particle *pp, int counts[26]) {
    int i, start, count;
    MPI_Request req;
    for (i = 0; i < 26; ++i) {
        start = ti.starts[i];
        count = counts[i];
        MC(m::Isend(pp + start, count * 6, MPI_FLOAT, ranks[i], t.btp1 + i, m::cart, &req));
        reqsendP.push_back(req);
    }
}
}
