namespace rex {

void recv_p(MPI_Comm cart, int ranks[26], int tags[26], x::TicketTags t) {
    for (int i = 0; i < 26; ++i) {
        int count = recv_counts[i];
        int expected = remote[i]->expected();

        remote[i]->pmessage.resize(max(1, count));
        remote[i]->preserve_resize(count);
        MPI_Status s;
        if (count > expected)
            MC(MPI_Recv(&remote[i]->pmessage.front() + expected, (count - expected) * 6, MPI_FLOAT, ranks[i], t.btp2 + tags[i], cart, &s));
        memcpy(remote[i]->hstate.D, &remote[i]->pmessage.front(), sizeof(Particle) * count);
    }

    postrecvC(cart, ranks, tags, t);
    for (int i = 0; i < 26; ++i) CC(cudaMemcpyAsync(remote[i]->dstate.D, remote[i]->hstate.D, sizeof(Particle) * remote[i]->hstate.S, H2D));
}


void recv_copy_bags() {
    float *recvbags[26];
    for (int i = 0; i < 26; ++i) recvbags[i] = (float *)local[i]->result->DP;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::recvbags, recvbags, sizeof(recvbags), 0, H2D));
}

void recv_f(std::vector<ParticlesWrap> w, x::TicketPack tp) {
    for (int i = 0; i < (int) w.size(); ++i) {
        ParticlesWrap it = w[i];
        if (it.n) {
            CC(cudaMemcpyToSymbolAsync(k_rex::g::starts,  tp.starts  + 27 * i, sizeof(int) * 27, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::g::counts,  tp.counts  + 26 * i, sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::g::offsets, tp.offsets + 26 * i, sizeof(int) * 26, 0, D2D));
            k_rex::unpack<<<16 * 14, 128>>>(/**/ (float *)it.f);
        }
    }
}
}
