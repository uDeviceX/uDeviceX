namespace rex {
void recvF(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    int i, tag;
    MPI_Request reqA;
    for (i = 0; i < 26; ++i) {
        tag = t.btf + tags[i];
        MC(l::m::Irecv(local[i]->ff->D, local[i]->ff->S * 3, MPI_FLOAT, ranks[i], tag, cart, &reqA));
        reqrecvA.push_back(reqA);
    }
}

void recvC(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    int i, tag;
    MPI_Request reqC;
    for (i = 0; i < 26; ++i) {
        tag = t.btc + tags[i];
        MC(l::m::Irecv(recv_counts + i, 1, MPI_INTEGER, ranks[i], tag, cart, &reqC));
        reqrecvC.push_back(reqC);
    }
}

void recvP1(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    int i, tag, n;
    MPI_Request reqP;
    for (i = 0; i < 26; ++i) {
        tag = t.btp1 + tags[i];
        n = recv_counts[i];
        remote[i]->pp.resize(n);
        MC(l::m::Irecv(&remote[i]->pp.front(), n * 6, MPI_FLOAT, ranks[i], tag, cart, &reqP));
        reqrecvP.push_back(reqP);
    }
}
}
