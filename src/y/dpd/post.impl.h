namespace dpd {
  
void post(MPI_Comm cart, Particle *pp, SendHalo* sendhalos[], int n) {
    dSync(); /* wait for fill_all */

    for (int i = 0; i < 26; ++i) {
        int nrequired = required_send_bag_size_host[i];

        sendhalos[i]->dbuf->S = nrequired;
        sendhalos[i]->hbuf->resize(nrequired);
        sendhalos[i]->scattered_entries->S = nrequired;
    }
    
    for (int i = 0; i < 26; ++i)
    if (sendhalos[i]->hbuf->S)
    cudaMemcpyAsync(sendhalos[i]->hbuf->D, sendhalos[i]->dbuf->D,
                    sizeof(Particle) * sendhalos[i]->hbuf->S,
                    H2H);
    dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */

    for (int i = 0, c = 0; i < 26; ++i)
    MC(l::m::Isend(sendhalos[i]->hcellstarts->D,
                   sendhalos[i]->hcellstarts->S, MPI_INTEGER, dstranks[i],
                   BT_CS_DPD + i, cart, sendcellsreq + c++));

    for (int i = 0, c = 0; i < 26; ++i)
    MC(l::m::Isend(&sendhalos[i]->hbuf->S, 1, MPI_INTEGER, dstranks[i],
                   BT_C_DPD + i, cart, sendcountreq + c++));

    for (int i = 0; i < 26; ++i) {
        int count = sendhalos[i]->hbuf->S;

        MC(l::m::Isend(sendhalos[i]->hbuf->D, count, Particle::datatype(),
                       dstranks[i], BT_P_DPD + i, cart, sendreq + i));
    }
}
}
