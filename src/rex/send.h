namespace rex {
void sendF(MPI_Comm cart, int ranks[26], x::TicketTags t) {
    dSync();
    reqsendA.resize(26);
    for (int i = 0; i < 26; ++i) MC(l::m::Isend(remote[i]->result.D, remote[i]->result.S * 3, MPI_FLOAT, ranks[i], t.btf + i, cart, &reqsendA[i]));
}

void sendP(MPI_Comm cart, int dranks[26], x::TicketTags t, x::TicketPinned ti) {
    reqsendC.resize(26);
    for (int i = 0; i < 26; ++i)
        MC(l::m::Isend(send_counts + i, 1, MPI_INTEGER, dranks[i], t.btc + i, cart, &reqsendC[i]));

    for (int i = 0; i < 26; ++i) {
        int start = ti.tstarts[i];
        int count = send_counts[i];
        int expected = local[i]->expected();
        
        MPI_Request reqP;
        MC(l::m::Isend(host_packbuf + start, expected * 6, MPI_FLOAT, dranks[i], t.btp1 + i, cart, &reqP));
        reqsendP.push_back(reqP);
        
        if (count > expected) {
            MPI_Request reqP2;
            MC(l::m::Isend(host_packbuf + start + expected, (count - expected) * 6, MPI_FLOAT, dranks[i], t.btp2 + i, cart, &reqP2));
            reqsendP.push_back(reqP2);
        }
    }
}
}
