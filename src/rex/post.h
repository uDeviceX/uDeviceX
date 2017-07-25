namespace rex {
void post_waitC() { _wait(reqsendC); }

bool post_pre(MPI_Comm cart, int dranks[26], int tags[26], x::TicketTags t) {
    bool packingfailed;
    int i;

    if (cnt == 0) _postrecvC(cart, dranks, tags, t);
    else post_waitC();

    for (i = 0; i < 26; ++i) send_counts[i] = host_packstotalcount->D[i];
    packingfailed = false;
    for (i = 0; i < 26; ++i)
        packingfailed |= send_counts[i] > local[i]->capacity();
    return packingfailed;
}

void post_resize() {
    int newcapacities[26];
    int *newindices[26];
    int i;

    for (i = 0; i < 26; ++i) local[i]->resize(send_counts[i]);
    for (i = 0; i < 26; ++i) newcapacities[i] = local[i]->capacity();
    CC(cudaMemcpyToSymbolAsync(k_rex::ccapacities, newcapacities,
                               sizeof(newcapacities), 0,
                               H2D));
    for (i = 0; i < 26; ++i) newindices[i] = local[i]->scattered_indices->D;
    CC(cudaMemcpyToSymbolAsync(k_rex::scattered_indices, newindices,
                               sizeof(newindices), 0, H2D));
}

void local_resize() {
    int i;
    for (i = 0; i < 26; ++i) local[i]->resize(send_counts[i]);
}

void post_p(MPI_Comm cart, int dranks[26], int tags[26], x::TicketTags t) {
    // consolidate the packing
    if (cnt == 0)
        _postrecvP(cart, dranks, tags, t);
    else
        _wait(reqsendP);
    
    if (host_packstotalstart->D[26]) {
        CC(cudaMemcpyAsync(host_packbuf->D, packbuf->D,
                           sizeof(Particle) * host_packstotalstart->D[26],
                           H2H));
    }
    dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */

    reqsendC.resize(26);

    for (int i = 0; i < 26; ++i)
        MC(l::m::Isend(send_counts + i, 1, MPI_INTEGER, dranks[i],
                       t.btc + i, cart, &reqsendC[i]));

    for (int i = 0; i < 26; ++i) {
        int start = host_packstotalstart->D[i];
        int count = send_counts[i];
        int expected = local[i]->expected();
        
        MPI_Request reqP;
        MC(l::m::Isend(host_packbuf->D + start, expected * 6, MPI_FLOAT,
                       dranks[i], t.btp1 + i, cart, &reqP));
        reqsendP.push_back(reqP);
        
        if (count > expected) {
            MPI_Request reqP2;
            MC(l::m::Isend(host_packbuf->D + start + expected,
                           (count - expected) * 6, MPI_FLOAT, dranks[i],
                           t.btp2 + i, cart, &reqP2));
            
            reqsendP.push_back(reqP2);
        }
    }
}
}
