namespace rex {
void _wait(std::vector<MPI_Request> &v) {
    MPI_Status statuses[v.size()];
    if (v.size()) MC(l::m::Waitall(v.size(), &v.front(), statuses));
    v.clear();
}

void _postrecvC() {
    for (int i = 0; i < 26; ++i) {
        MPI_Request reqC;
        MC(l::m::Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i],
                     btc + recv_tags[i], cart, &reqC));
        reqrecvC.push_back(reqC);
    }
}

void _postrecvP() {
    for (int i = 0; i < 26; ++i) {
        MPI_Request reqP;
        remote[i]->pmessage.resize(remote[i]->expected());
        MC(l::m::Irecv(&remote[i]->pmessage.front(), remote[i]->expected() * 6,
                     MPI_FLOAT, dstranks[i], btp1 + recv_tags[i],
                     cart, &reqP));
        reqrecvP.push_back(reqP);
    }
}

void _postrecvA() {
    for (int i = 0; i < 26; ++i) {
        MPI_Request reqA;

        MC(l::m::Irecv(local[i]->result->D, local[i]->result->S * 3,
                     MPI_FLOAT, dstranks[i], btf + recv_tags[i],
                     cart, &reqA));
        reqrecvA.push_back(reqA);
    }
}

void post_p(std::vector<ParticlesWrap> w) {
    // consolidate the packing
    {
        dSync();

        if (iterationcount == 0)
        _postrecvC();
        else
        _wait(reqsendC);

        for (int i = 0; i < 26; ++i) send_counts[i] = host_packstotalcount->D[i];

        bool packingfailed = false;

        for (int i = 0; i < 26; ++i)
        packingfailed |= send_counts[i] > local[i]->capacity();

        if (packingfailed) {
            for (int i = 0; i < 26; ++i) local[i]->resize(send_counts[i]);

            int newcapacities[26];
            for (int i = 0; i < 26; ++i) newcapacities[i] = local[i]->capacity();

            CC(cudaMemcpyToSymbolAsync(k_rex::ccapacities, newcapacities,
                                       sizeof(newcapacities), 0,
                                       H2D));

            int *newindices[26];
            for (int i = 0; i < 26; ++i) newindices[i] = local[i]->scattered_indices->D;

            CC(cudaMemcpyToSymbolAsync(k_rex::scattered_indices, newindices,
                                       sizeof(newindices), 0, H2D));

            _adjust_packbuffers();

            _pack_attempt(w);

            dSync(); /* was CC(cudaStreamSynchronize(stream)); */
        }

        for (int i = 0; i < 26; ++i) local[i]->resize(send_counts[i]);

        _postrecvA();

        if (iterationcount == 0) {
            _postrecvP();
        } else
        _wait(reqsendP);

        if (host_packstotalstart->D[26]) {
            CC(cudaMemcpyAsync(host_packbuf->D, packbuf->D,
                               sizeof(Particle) * host_packstotalstart->D[26],
                               H2H));
        }
        dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */
    }

    // post the sending of the packs
    {
        reqsendC.resize(26);

        for (int i = 0; i < 26; ++i)
        MC(l::m::Isend(send_counts + i, 1, MPI_INTEGER, dstranks[i],
                     btc + i, cart, &reqsendC[i]));

        for (int i = 0; i < 26; ++i) {
            int start = host_packstotalstart->D[i];
            int count = send_counts[i];
            int expected = local[i]->expected();

            MPI_Request reqP;
            MC(l::m::Isend(host_packbuf->D + start, expected * 6, MPI_FLOAT,
                         dstranks[i], btp1 + i, cart, &reqP));
            reqsendP.push_back(reqP);

            if (count > expected) {
                MPI_Request reqP2;
                MC(l::m::Isend(host_packbuf->D + start + expected,
                             (count - expected) * 6, MPI_FLOAT, dstranks[i],
                             btp2 + i, cart, &reqP2));

                reqsendP.push_back(reqP2);
            }
        }
    }
}

void recv_p(std::vector<ParticlesWrap> w) {
    if (w.size() == 0) return;

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
                    (count - expected) * 6, MPI_FLOAT, dstranks[i],
                    btp2 + recv_tags[i], cart, &status));

        memcpy(remote[i]->hstate.D, &remote[i]->pmessage.front(),
               sizeof(Particle) * count);
    }

    _postrecvC();

    for (int i = 0; i < 26; ++i)
    CC(cudaMemcpyAsync(remote[i]->dstate.D, remote[i]->hstate.D,
                       sizeof(Particle) * remote[i]->hstate.S,
                       H2D));
}

void halo() {
    if (iterationcount) _wait(reqsendA);

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

void post_f(std::vector<ParticlesWrap> w) {
    if (w.size() == 0) return;

    dSync(); /* was cudaEventSynchronize() */

    reqsendA.resize(26);
    for (int i = 0; i < 26; ++i)
    MC(l::m::Isend(remote[i]->result.D, remote[i]->result.S * 3,
                 MPI_FLOAT, dstranks[i], btf + i, cart,
                 &reqsendA[i]));
}

void recv_f(std::vector<ParticlesWrap> w) {
    if (w.size() == 0) return;

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
