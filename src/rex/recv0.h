namespace rex {
void recv0(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    int i, tag, n;
    MPI_Request reqP;
    Particle *p;
    n = MAX_PART_NUM;
    for (i = 0; i < 26; ++i) {
        tag = t.btp1 + tags[i];
        remote[i]->pp.resize(n);
        p = remote[i]->pp.data();
        MC(l::m::Irecv(p, n, datatype::particle, ranks[i], tag, cart, &reqP));
        reqrecvP.push_back(reqP);
    }
}
}
