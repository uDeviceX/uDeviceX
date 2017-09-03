namespace rex {
void recvF(int ranks[26], int tags[26], x::TicketTags t, int counts[26], LFrag *local) {
    int i, count, tag;
    MPI_Request reqA;
    for (i = 0; i < 26; ++i) {
        tag = t.btf + tags[i];
        count = counts[i];
        MC(m::Irecv(local[i].ff_pi, 3 * count, MPI_FLOAT, ranks[i], tag, m::cart, &reqA));
        reqrecvA.push_back(reqA);
    }
}

void recvC(int ranks[26], int tags[26], x::TicketTags t, int counts[26]) {
    int i, tag;
    MPI_Request reqC;
    for (i = 0; i < 26; ++i) {
        tag = t.btc + tags[i];
        MC(m::Irecv(counts + i, 1, MPI_INTEGER, ranks[i], tag, m::cart, &reqC));
        reqrecvC.push_back(reqC);
    }
}

void recvP(int ranks[26], int tags[26], x::TicketTags t, int counts[26], Pap26 PP_pi) {
    int i, tag, n;
    MPI_Request reqP;
    Particle *p;
    for (i = 0; i < 26; ++i) {
        tag = t.btp1 + tags[i];
        n = counts[i];
        p = PP_pi.d[i];
        MC(m::Irecv(p, n * 6, MPI_FLOAT, ranks[i], tag, m::cart, &reqP));
        reqrecvP.push_back(reqP);
    }
}
}
