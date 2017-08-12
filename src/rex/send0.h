namespace rex {
void send0(MPI_Comm cart, int ranks[26], x::TicketTags t, x::TicketPinned ti, Particle *pp) {
    int i, start, count;
    MPI_Request req;
    for (i = 0; i < 26; ++i) {
        start = ti.tstarts[i];
        count = send_counts[i];
        MC(l::m::Isend(pp + start, count * 6, MPI_FLOAT, ranks[i], t.btp1 + i, cart, &req));
        reqsendP.push_back(req);
    }
}
}
