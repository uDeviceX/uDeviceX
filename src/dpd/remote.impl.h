namespace DPD {
void init1() {
    for (int i = 0; i < 26; ++i) {
        int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

        int coordsneighbor[3];
        for (int c = 0; c < 3; ++c)
        coordsneighbor[c] = (m::coords[c] + d[c] + m::dims[c]) % m::dims[c];

        int indx[3];
        for (int c = 0; c < 3; ++c)
        indx[c] = min(m::coords[c], coordsneighbor[c]) * m::dims[c] +
            max(m::coords[c], coordsneighbor[c]);

        int interrank_seed_base =
            indx[0] + m::dims[0] * m::dims[0] * (indx[1] + m::dims[1] * m::dims[1] * indx[2]);

        int interrank_seed_offset;

        {
            bool isplus =
                d[0] + d[1] + d[2] > 0 ||
                d[0] + d[1] + d[2] == 0 &&
                (d[0] > 0 || d[0] == 0 && (d[1] > 0 || d[1] == 0 && d[2] > 0));

            int mysign = 2 * isplus - 1;

            int v[3] = {1 + mysign * d[0], 1 + mysign * d[1], 1 + mysign * d[2]};

            interrank_seed_offset = v[0] + 3 * (v[1] + 3 * v[2]);
        }

        int interrank_seed = interrank_seed_base + interrank_seed_offset;

        interrank_trunks[i] = new l::rnd::d::KISS(390 + interrank_seed, interrank_seed + 615, 12309, 23094);
        int dstrank = dstranks[i];

        if (dstrank != m::rank)
        interrank_masks[i] = min(dstrank, m::rank) == m::rank;
        else {
            int alter_ego =
                (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
            interrank_masks[i] = min(i, alter_ego) == i;
        }
    }
}

void local_interactions(float4 *zip0, ushort4 *zip1, int n, int *start, int *count, /**/
                        Force *ff) {
    if (n > 0)
      forces_dpd_cuda_nohost(zip0, zip1, (float*)ff, n,
			     start, count, 1, XS, YS, ZS, 1. / sqrt(dt),
			     local_trunk->get_float());
}

void remote_interactions(int n, Force *a) {
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

    BipsBatch::interactions(1. / sqrt(dt), infos,
                            (float *)a, n);


}

void init0() {
    firstpost = true;
    nactive = 26;
    safety_factor =
        getenv("HEX_COMM_FACTOR") ? atof(getenv("HEX_COMM_FACTOR")) : 1.2;

    MC(l::m::Comm_dup(m::cart, &cart));
  
    for (int i = 0; i < 26; ++i) {
        int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
        recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
        int coordsneighbor[3];
        for (int c = 0; c < 3; ++c) coordsneighbor[c] = m::coords[c] + d[c];
        MC(l::m::Cart_rank(cart, coordsneighbor, dstranks + i));
        halosize[i].x = d[0] != 0 ? 1 : XS;
        halosize[i].y = d[1] != 0 ? 1 : YS;
        halosize[i].z = d[2] != 0 ? 1 : ZS;

        int nhalocells = halosize[i].x * halosize[i].y * halosize[i].z;

        int estimate = numberdensity * safety_factor * nhalocells;
        estimate = 32 * ((estimate + 31) / 32);

        recvhalos[i]->setup(estimate, nhalocells);
        sendhalos[i]->setup(estimate, nhalocells);
    }

    CC(cudaHostAlloc((void **)&required_send_bag_size_host, sizeof(int) * 26,
                     cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&required_send_bag_size,
                                required_send_bag_size_host, 0));
    CC(cudaEventCreateWithFlags(&evfillall, cudaEventDisableTiming));
    CC(cudaEventCreateWithFlags(&evdownloaded,
                                cudaEventDisableTiming | cudaEventBlockingSync));
}

void ini() {
    local_trunk = new l::rnd::d::KISS(0, 0, 0, 0);
    for (int i = 0; i < 26; i++) recvhalos[i] = new RecvHalo;
    for (int i = 0; i < 26; i++) sendhalos[i] = new SendHalo;

    init0();
    init1();
}

void _pack_all(Particle *p, int n, bool update_baginfos) {
    if (update_baginfos) {
        static phalo::SendBagInfo baginfos[26];
        for (int i = 0; i < 26; ++i) {
            baginfos[i].start_src = sendhalos[i]->tmpstart->D;
            baginfos[i].count_src = sendhalos[i]->tmpcount->D;
            baginfos[i].start_dst = sendhalos[i]->dcellstarts->D;
            baginfos[i].bagsize = sendhalos[i]->dbuf->C;
            baginfos[i].scattered_entries = sendhalos[i]->scattered_entries->D;
            baginfos[i].dbag = sendhalos[i]->dbuf->D;
            baginfos[i].hbag = sendhalos[i]->hbuf->D;
        }
        CC(cudaMemcpyToSymbolAsync(phalo::baginfos, baginfos, sizeof(baginfos), 0, H2D));
    }

    if (ncells)
      phalo::fill_all<<<(ncells + 1) / 2, 32>>>(p, n, required_send_bag_size);
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

void pack(Particle *p, int n, int *cellsstart, int *cellscount) {

    nlocal = n;
    if (firstpost) {
        {
            static int cellpackstarts[27];
            cellpackstarts[0] = 0;
            for (int i = 0, s = 0; i < 26; ++i)
            cellpackstarts[i + 1] =
                (s += sendhalos[i]->dcellstarts->S * (sendhalos[i]->expected > 0));
            ncells = cellpackstarts[26];
            CC(cudaMemcpyToSymbol(phalo::cellpackstarts, cellpackstarts,
                                  sizeof(cellpackstarts), 0, H2D));
        }

        {
            static phalo::CellPackSOA cellpacks[26];
            for (int i = 0; i < 26; ++i) {
                cellpacks[i].start = sendhalos[i]->tmpstart->D;
                cellpacks[i].count = sendhalos[i]->tmpcount->D;
                cellpacks[i].enabled = sendhalos[i]->expected > 0;
                cellpacks[i].scan = sendhalos[i]->dcellstarts->D;
                cellpacks[i].size = sendhalos[i]->dcellstarts->S;
            }
            CC(cudaMemcpyToSymbol(phalo::cellpacks, cellpacks,
                                  sizeof(cellpacks), 0, H2D));
        }
    }

    if (ncells)
    phalo::
        count_all<<<k_cnf(ncells)>>>(cellsstart, cellscount, ncells);

    phalo::scan_diego<32><<<26, 32 * 32>>>();

    if (firstpost)
    post_expected_recv();
    else {
        MPI_Status statuses[26 * 2];
        MC(l::m::Waitall(nactive, sendcellsreq, statuses));
        MC(l::m::Waitall(nsendreq, sendreq, statuses));
        MC(l::m::Waitall(nactive, sendcountreq, statuses));
    }

    if (firstpost) {
        {
            static int *srccells[26];
            for (int i = 0; i < 26; ++i) srccells[i] = sendhalos[i]->dcellstarts->D;

            CC(cudaMemcpyToSymbol(phalo::srccells, srccells, sizeof(srccells),
                                  0, H2D));

            static int *dstcells[26];
            for (int i = 0; i < 26; ++i)
            dstcells[i] = sendhalos[i]->hcellstarts->DP;

            CC(cudaMemcpyToSymbol(phalo::dstcells, dstcells, sizeof(dstcells),
                                  0, H2D));
        }

        {
            static int *srccells[26];
            for (int i = 0; i < 26; ++i) srccells[i] = recvhalos[i]->hcellstarts->DP;
            CC(cudaMemcpyToSymbol(phalo::srccells, srccells, sizeof(srccells),
                                  sizeof(srccells), H2D));

            static int *dstcells[26];
            for (int i = 0; i < 26; ++i) dstcells[i] = recvhalos[i]->dcellstarts->D;
            CC(cudaMemcpyToSymbol(phalo::dstcells, dstcells, sizeof(dstcells),
                                  sizeof(dstcells), H2D));
        }
    }

    if (ncells) phalo::copycells<0><<<k_cnf(ncells)>>>(ncells);
    _pack_all(p, n, firstpost);

}

void post(Particle *p, int n) {
    {
        CC(cudaEventSynchronize(evfillall));

        bool succeeded = true;
        for (int i = 0; i < 26; ++i) {
            int nrequired = required_send_bag_size_host[i];
            bool failed_entry = nrequired > sendhalos[i]->dbuf->C;

            if (failed_entry) {
                sendhalos[i]->dbuf->resize(nrequired);
                // sendhalos[i].hbuf.resize(nrequired);
                sendhalos[i]->scattered_entries->resize(nrequired);
                succeeded = false;
            }
        }

        if (!succeeded) {
            _pack_all(p, n, true);

            CC(cudaEventSynchronize(evfillall));
        }

        for (int i = 0; i < 26; ++i) {
            int nrequired = required_send_bag_size_host[i];

            sendhalos[i]->dbuf->S = nrequired;
            sendhalos[i]->hbuf->resize(nrequired);
            sendhalos[i]->scattered_entries->S = nrequired;
        }
    }

    for (int i = 0; i < 26; ++i)
    if (sendhalos[i]->hbuf->S)
    cudaMemcpyAsync(sendhalos[i]->hbuf->D, sendhalos[i]->dbuf->D,
                    sizeof(Particle) * sendhalos[i]->hbuf->S,
                    H2H);
    dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */

    {
        for (int i = 0, c = 0; i < 26; ++i)
        if (sendhalos[i]->expected)
        MC(l::m::Isend(sendhalos[i]->hcellstarts->D,
                     sendhalos[i]->hcellstarts->S, MPI_INTEGER, dstranks[i],
                     BT_CS_DPD + i, cart, sendcellsreq + c++));

        for (int i = 0, c = 0; i < 26; ++i)
        if (sendhalos[i]->expected)
        MC(l::m::Isend(&sendhalos[i]->hbuf->S, 1, MPI_INTEGER, dstranks[i],
                     BT_C_DPD + i, cart, sendcountreq + c++));

        nsendreq = 0;

        for (int i = 0; i < 26; ++i) {
            int expected = sendhalos[i]->expected;

            if (expected == 0) continue;

            int count = sendhalos[i]->hbuf->S;

            MC(l::m::Isend(sendhalos[i]->hbuf->D, expected, Particle::datatype(),
                         dstranks[i], BT_P_DPD + i, cart, sendreq + nsendreq));

            ++nsendreq;

            if (count > expected) {

                int difference = count - expected;

                int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
                printf("extra message from rank %d to rank %d in the direction of %d "
                       "%d %d! difference %d, expected is %d\n",
                       m::rank, dstranks[i], d[0], d[1], d[2], difference, expected);

                MC(l::m::Isend(sendhalos[i]->hbuf->D + expected, difference,
                             Particle::datatype(), dstranks[i], BT_P2_DPD + i,
                             cart, sendreq + nsendreq));
                ++nsendreq;
            }
        }
    }
    firstpost = false;
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
