namespace rex {
bool post_pre(MPI_Comm cart, int dstranks[26]) {
    bool packingfailed;
    int i;

    dSync();
    if (iterationcount == 0) _postrecvC(cart, dstranks);
    else _wait(reqsendC);

    for (i = 0; i < 26; ++i) send_counts[i] = host_packstotalcount->D[i];
    packingfailed = false;
    for (i = 0; i < 26; ++i)
        packingfailed |= send_counts[i] > local[i]->capacity();
    return packingfailed;
}

void post_p(MPI_Comm cart, int dstranks[26], std::vector<ParticlesWrap> w) {
    // consolidate the packing
    {
        bool packingfailed;
        packingfailed = post_pre(cart, dstranks);

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

        _postrecvA(cart, dstranks);

        if (iterationcount == 0) {
            _postrecvP(cart, dstranks);
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
}
