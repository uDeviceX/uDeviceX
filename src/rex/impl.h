namespace rex {
void _wait(std::vector<MPI_Request> &v) {
    MPI_Status statuses[v.size()];
    if (v.size()) MC(l::m::Waitall(v.size(), &v.front(), statuses));
    v.clear();
}

void _postrecvC(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    for (int i = 0; i < 26; ++i) {
        MPI_Request reqC;
        MC(l::m::Irecv(recv_counts + i, 1, MPI_INTEGER, ranks[i],
                     t.btc + tags[i], cart, &reqC));
        reqrecvC.push_back(reqC);
    }
}

void _postrecvP(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    for (int i = 0; i < 26; ++i) {
        MPI_Request reqP;
        remote[i]->pmessage.resize(remote[i]->expected());
        MC(l::m::Irecv(&remote[i]->pmessage.front(), remote[i]->expected() * 6,
                     MPI_FLOAT, ranks[i], t.btp1 + tags[i],
                     cart, &reqP));
        reqrecvP.push_back(reqP);
    }
}

void _postrecvA(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    for (int i = 0; i < 26; ++i) {
        MPI_Request reqA;

        MC(l::m::Irecv(local[i]->result->D, local[i]->result->S * 3,
                     MPI_FLOAT, ranks[i], t.btf + tags[i],
                     cart, &reqA));
        reqrecvA.push_back(reqA);
    }
}


void recv_p(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    _wait(reqrecvC);
    _wait(reqrecvP);

    for (int i = 0; i < 26; ++i) {
        int count = recv_counts[i];
        int expected = remote[i]->expected();

        remote[i]->pmessage.resize(max(1, count));
        remote[i]->preserve_resize(count);
        MPI_Status status;

        if (count > expected)
            MC(MPI_Recv(&remote[i]->pmessage.front() + expected,
                        (count - expected) * 6, MPI_FLOAT, ranks[i],
                        t.btp2 + tags[i], cart, &status));

        memcpy(remote[i]->hstate.D, &remote[i]->pmessage.front(),
               sizeof(Particle) * count);
    }

    _postrecvC(cart, ranks, tags, t);

    for (int i = 0; i < 26; ++i)
        CC(cudaMemcpyAsync(remote[i]->dstate.D, remote[i]->hstate.D,
                       sizeof(Particle) * remote[i]->hstate.S,
                       H2D));
}

void halo() {
    if (cnt) _wait(reqsendA);

    ParticlesWrap halos[26];

    for (int i = 0; i < 26; ++i)
        halos[i] = ParticlesWrap(remote[i]->dstate.D, remote[i]->dstate.S,
                                 remote[i]->result.DP);

    dSync(); /* was CC(cudaStreamSynchronize(uploadstream)); */

    /** here was visitor  **/
    if (fsiforces)     fsi::halo(halos);
    if (contactforces) cnt::halo(halos);
    /***********************/
    for (int i = 0; i < 26; ++i) local[i]->update();
}

void post_f(MPI_Comm cart, int ranks[26], x::TicketTags t) {
    dSync(); /* was cudaEventSynchronize() */

    reqsendA.resize(26);
    for (int i = 0; i < 26; ++i)
    MC(l::m::Isend(remote[i]->result.D, remote[i]->result.S * 3,
                 MPI_FLOAT, ranks[i], t.btf + i, cart,
                 &reqsendA[i]));
}

void recv_f(std::vector<ParticlesWrap> w) {
    {
        float *recvbags[26];

        for (int i = 0; i < 26; ++i) recvbags[i] = (float *)local[i]->result->DP;

        CC(cudaMemcpyToSymbolAsync(k_rex::recvbags, recvbags, sizeof(recvbags),
                                   0, H2D));
    }

    _wait(reqrecvA);

    for (int i = 0; i < (int) w.size(); ++i) {
        ParticlesWrap it = w[i];

        if (it.n) {
            CC(cudaMemcpyToSymbolAsync(k_rex::cpaddedstarts,
                                       packsstart->D + 27 * i, sizeof(int) * 27, 0,
                                       D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::ccounts, packscount->D + 26 * i,
                                       sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::coffsets, packsoffset->D + 26 * i,
                                       sizeof(int) * 26, 0, D2D));

            k_rex::unpack<<<16 * 14, 128>>>(it.n, /**/ (float *)it.f);
        }

    }
}
}
