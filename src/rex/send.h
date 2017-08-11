namespace rex {
void sendF(MPI_Comm cart, int ranks[26], x::TicketTags t) {
    int i;
    reqsendA.resize(26);
    for (i = 0; i < 26; ++i)
        MC(l::m::Isend(remote[i]->ff.D, remote[i]->ff.S * 3, MPI_FLOAT, ranks[i], t.btf + i, cart, &reqsendA[i]));
}

void sendC(MPI_Comm cart, int dranks[26], x::TicketTags t) {
    int i;
    reqsendC.resize(26);
    for (i = 0; i < 26; ++i)
        MC(l::m::Isend(send_counts + i, 1, MPI_INTEGER, dranks[i], t.btc + i, cart, &reqsendC[i]));
}

void sendP12(MPI_Comm cart, int ranks[26], x::TicketTags t, x::TicketPinned ti, Particle *pp) {
    int i, start, count, expected;
    MPI_Request req;
    for (i = 0; i < 26; ++i) {
        start = ti.tstarts[i];
        count = send_counts[i];
        expected = lo::expected(local[i]);
        
        MC(l::m::Isend(pp + start, expected * 6, MPI_FLOAT, ranks[i], t.btp1 + i, cart, &req));
        reqsendP.push_back(req);
        
        if (count > expected) {
            MC(l::m::Isend(pp + start + expected,
                           (count - expected) * 6,
                           MPI_FLOAT,
                           ranks[i], t.btp2 + i, cart, &req));
            reqsendP.push_back(req);
        }
    }
}
}
