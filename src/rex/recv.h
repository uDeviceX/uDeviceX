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

void recvP(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    int i, tag, n;
    MPI_Request reqP;
    for (i = 0; i < 26; ++i) {
        tag = t.btp1 + tags[i];
        n = remote[i]->expected();
        remote[i]->pmessage.resize(n);
        MC(l::m::Irecv(&remote[i]->pmessage.front(), n * 6, MPI_FLOAT, ranks[i], tag, cart, &reqP));
        reqrecvP.push_back(reqP);
    }
}

void recvM(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    int tag, rank;
    int i, count, expected, n;
    MPI_Status s;
    for (i = 0; i < 26; ++i) {
        count = recv_counts[i];
        expected = remote[i]->expected();
        remote[i]->pmessage.resize(max(1, count));
        remote[i]->resize(count);
        n = count - expected;
        if (n > 0) {
            tag = t.btp2 + tags[i];
            rank = ranks[i];
            MC(l::m::Recv(&remote[i]->pmessage.front() + expected, n * 6, MPI_FLOAT, rank, tag, cart, &s));
        }
    }
}
}
