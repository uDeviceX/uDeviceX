void recv0(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    int i, tag, n;
    MPI_Request reqP;
    for (i = 0; i < 26; ++i) {
        tag = t.btp1 + tags[i];
        n = re::expected(remote[i]);
        remote[i]->pmessage.resize(n);
        MC(l::m::Irecv(&remote[i]->pmessage.front(), n * 6, MPI_FLOAT, ranks[i], tag, cart, &reqP));
        reqrecvP.push_back(reqP);
    }
}
