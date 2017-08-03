namespace rex {
void recvF(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    for (int i = 0; i < 26; ++i) {
        MPI_Request reqA;
        MC(l::m::Irecv(local[i]->result->D, local[i]->result->S * 3, MPI_FLOAT, ranks[i], t.btf + tags[i], cart, &reqA));
        reqrecvA.push_back(reqA);
    }
}

void recvC(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    for (int i = 0; i < 26; ++i) {
        MPI_Request reqC;
        MC(l::m::Irecv(recv_counts + i, 1, MPI_INTEGER, ranks[i], t.btc + tags[i], cart, &reqC));
        reqrecvC.push_back(reqC);
    }
}

void recvP(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    for (int i = 0; i < 26; ++i) {
        MPI_Request reqP;
        remote[i]->pmessage.resize(remote[i]->expected());
        MC(l::m::Irecv(&remote[i]->pmessage.front(), remote[i]->expected() * 6, MPI_FLOAT, ranks[i], t.btp1 + tags[i], cart, &reqP));
        reqrecvP.push_back(reqP);
    }
}

}
