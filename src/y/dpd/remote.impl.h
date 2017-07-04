namespace dpd {
void _pack_all(Particle *p, int n, bool update_baginfos) {
    if (update_baginfos) {
        static k_halo::SendBagInfo baginfos[26];
        for (int i = 0; i < 26; ++i) {
            baginfos[i].start_src = sendhalos[i]->tmpstart->D;
            baginfos[i].count_src = sendhalos[i]->tmpcount->D;
            baginfos[i].start_dst = sendhalos[i]->dcellstarts->D;
            baginfos[i].bagsize = sendhalos[i]->dbuf->C;
            baginfos[i].scattered_entries = sendhalos[i]->scattered_entries->D;
            baginfos[i].dbag = sendhalos[i]->dbuf->D;
            baginfos[i].hbag = sendhalos[i]->hbuf->D;
        }
        CC(cudaMemcpyToSymbolAsync(k_halo::baginfos, baginfos, sizeof(baginfos), 0, H2D));
    }

    if (ncells)
      k_halo::fill_all<<<(ncells + 1) / 2, 32>>>(p, n, required_send_bag_size);
    CC(cudaEventRecord(evfillall));
}

void post_expected_recv() {
    for (int i = 0, c = 0; i < 26; ++i) {
        if (recvhalos[i]->expected)
        MC(l::m::Irecv(recvhalos[i]->hbuf->D, recvhalos[i]->expected,
                     Particle::datatype(), dstranks[i], BT_P_DPD + recv_tags[i],
                     cart, recvreq + c++));
    }
    for (int i = 0, c = 0; i < 26; ++i)
    if (recvhalos[i]->expected)
    MC(l::m::Irecv(recvhalos[i]->hcellstarts->D,
                 recvhalos[i]->hcellstarts->S, MPI_INTEGER, dstranks[i],
                 BT_CS_DPD + recv_tags[i], cart, recvcellsreq + c++));

    for (int i = 0, c = 0; i < 26; ++i)
    if (recvhalos[i]->expected)
    MC(l::m::Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i],
                 BT_C_DPD + recv_tags[i], cart, recvcountreq + c++));
    else
    recv_counts[i] = 0;
}

void _cancel_recv() {
    if (!firstpost) {
        {
            MPI_Status statuses[26 * 2];
            MC(l::m::Waitall(nactive, sendcellsreq, statuses));
            MC(l::m::Waitall(nsendreq, sendreq, statuses));
            MC(l::m::Waitall(nactive, sendcountreq, statuses));
        }

        for (int i = 0; i < nactive; ++i) MC(MPI_Cancel(recvreq + i));
        for (int i = 0; i < nactive; ++i) MC(MPI_Cancel(recvcellsreq + i));
        for (int i = 0; i < nactive; ++i) MC(MPI_Cancel(recvcountreq + i));
        firstpost = true;
    }
}

void fin() {
    CC(cudaFreeHost(required_send_bag_size));
    MC(l::m::Comm_free(&cart));
    _cancel_recv();
    CC(cudaEventDestroy(evfillall));
    CC(cudaEventDestroy(evdownloaded));

    for (int i = 1; i < 26; i++) delete interrank_trunks[i];
    delete local_trunk;
    for (int i = 0; i < 26; i++) delete recvhalos[i];
    for (int i = 0; i < 26; i++) delete sendhalos[i];
}
}
