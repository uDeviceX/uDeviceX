namespace dpd {
void wait_recv() {
    MPI_Status statuses[26];
    MC(l::m::Waitall(26, recvreq, statuses));
    MC(l::m::Waitall(26, recvcellsreq, statuses));
    MC(l::m::Waitall(26, recvcountreq, statuses));
}

void recv(MPI_Comm cart, RecvHalo* recvhalos[]) {
    for (int i = 0; i < 26; ++i) {
        int count = recv_counts[i];

        recvhalos[i]->hbuf->resize(count);
        recvhalos[i]->dbuf->resize(count);
    }

    for (int i = 0; i < 26; ++i)
    CC(cudaMemcpyAsync(recvhalos[i]->dbuf->D, recvhalos[i]->hbuf->D,
                       sizeof(Particle) * recvhalos[i]->hbuf->S,
                       H2D));
    
    for (int i = 0; i < 26; ++i)
    CC(cudaMemcpyAsync(recvhalos[i]->dcellstarts->D,
                       recvhalos[i]->hcellstarts->D,
                       sizeof(int) * recvhalos[i]->hcellstarts->S,
                       H2D));
}
}
