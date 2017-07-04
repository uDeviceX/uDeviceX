namespace dpd {
void flocal(float4 *zip0, ushort4 *zip1, int n, int *start, int *count, /**/
                        Force *ff) {
    if (n > 0)
      flocal0(zip0, zip1, (float*)ff, n,
	      start, count, 1, XS, YS, ZS, local_trunk->get_float());
}

void fremote(int n, Force *a) {
    static BipsBatch::BatchInfo infos[26];

    for (int i = 0; i < 26; ++i) {
        int dx = (i + 2) % 3 - 1;
        int dy = (i / 3 + 2) % 3 - 1;
        int dz = (i / 9 + 2) % 3 - 1;

        int m0 = 0 == dx;
        int m1 = 0 == dy;
        int m2 = 0 == dz;

        BipsBatch::BatchInfo entry = {
            (float *)sendhalos[i]->dbuf->D,
            (float2 *)recvhalos[i]->dbuf->D,
            interrank_trunks[i]->get_float(),
            sendhalos[i]->dbuf->S,
            recvhalos[i]->dbuf->S,
            interrank_masks[i],
            recvhalos[i]->dcellstarts->D,
            sendhalos[i]->scattered_entries->D,
            dx,
            dy,
            dz,
            1 + m0 * (XS - 1),
            1 + m1 * (YS - 1),
            1 + m2 * (ZS - 1),
            (BipsBatch::HaloType)(abs(dx) + abs(dy) + abs(dz))};

        infos[i] = entry;
    }

    BipsBatch::interactions(infos, (float *)a, n);
}

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

void recv() {

    {
        MPI_Status statuses[26];

        MC(l::m::Waitall(nactive, recvreq, statuses));
        MC(l::m::Waitall(nactive, recvcellsreq, statuses));
        MC(l::m::Waitall(nactive, recvcountreq, statuses));
    }

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


    post_expected_recv();
}

int nof_sent_particles() {
    int s = 0;
    for (int i = 0; i < 26; ++i) s += sendhalos[i]->hbuf->S;
    return s;
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
