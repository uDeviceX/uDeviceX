namespace rex {
void sendF(MPI_Comm cart, int ranks[26], x::TicketTags t) {
    int i, n;
    reqsendA.resize(26);
    for (i = 0; i < 26; ++i) {
        n = recv_counts[i];
        MC(l::m::Isend(remote[i].ff_pi, 3 * n, MPI_FLOAT, ranks[i], t.btf + i, cart, &reqsendA[i]));
    }
}

void sendC(MPI_Comm cart, int dranks[26], x::TicketTags t, int counts[26]) {
    int i;
    reqsendC.resize(26);
    for (i = 0; i < 26; ++i)
        MC(l::m::Isend(counts + i, 1, MPI_INTEGER, dranks[i], t.btc + i, cart, &reqsendC[i]));
}

void sendP(MPI_Comm cart, int ranks[26], x::TicketTags t, x::TicketPinned ti, Particle *pp, int counts[26]) {
    int i, start, count;
    MPI_Request req;
    for (i = 0; i < 26; ++i) {
        start = ti.starts[i];
        count = counts[i];
        MC(l::m::Isend(pp + start, count * 6, MPI_FLOAT, ranks[i], t.btp1 + i, cart, &req));
        reqsendP.push_back(req);
    }
}
}
