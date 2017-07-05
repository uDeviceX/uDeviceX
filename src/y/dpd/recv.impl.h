namespace dpd {
void wait_recv() {
  MPI_Status statuses[26];
  MC(l::m::Waitall(26, recvreq, statuses));
  MC(l::m::Waitall(26, recvcellsreq, statuses));
  MC(l::m::Waitall(26, recvcountreq, statuses));
}

void recv(MPI_Comm cart) {
    for (int i = 0; i < 26; ++i) {
        int count = recv_counts[i];
        int expected = recvhalos[i]->expected;
        int difference = count - expected;

        if (count <= expected) {
            recvhalos[i]->hbuf->resize(count);
            recvhalos[i]->dbuf->resize(count);
        } else {
            printf("RANK %d waiting for RECV-extra message: count %d expected %d "
                   "(difference %d) from rank %d\n",
                   m::rank, count, expected, difference, dstranks[i]);
            recvhalos[i]->hbuf->preserve_resize(count);
            recvhalos[i]->dbuf->resize(count);
            MPI_Status status;
            MPI_Recv(recvhalos[i]->hbuf->D + expected, difference,
                     Particle::datatype(), dstranks[i], BT_P2_DPD + recv_tags[i],
                     cart, &status);
        }
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
