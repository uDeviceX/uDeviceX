namespace odstr {
int pack_size(int code) { return send_sizes[code]; }
float pinned_data(int code, int entry) {return pinnedhost_sendbufs[code][entry]; }

void _waitall(MPI_Request * reqs, int n) {
    MPI_Status statuses[n];
    MC( MPI_Waitall(n, reqs, statuses) );
}

void init()  {
    failure = new PinnedHostBuffer<bool>(1);
    packsizes = new PinnedHostBuffer<int>(27);
    compressed_cellcounts = new DeviceBuffer<unsigned char>
        (XS * YS * ZS);
    remote_particles = new DeviceBuffer<Particle>;

    subindices_remote= new DeviceBuffer<uchar4>
        (1.5 * numberdensity * (XS * YS * ZS -
                                (XS - 2) * (YS - 2) * (ZS - 2)));
    subindices = new DeviceBuffer<uchar4>
        (1.5 * numberdensity * XS * YS * ZS);
    scattered_indices = new DeviceBuffer<uint>;

    nactiveneighbors  = 26; firstcall = true;
    MC(MPI_Comm_dup(m::cart, &cart));

    for(int i = 0; i < 27; ++i) {
        int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };
        recv_tags[i] = (3 - d[0]) % 3 + 3 * ((3 - d[1]) % 3 + 3 * ((3 - d[2]) % 3));
        int coordsneighbor[3];
        for(int c = 0; c < 3; ++c) coordsneighbor[c] = m::coords[c] + d[c];
        MC( MPI_Cart_rank(cart, coordsneighbor, neighbor_ranks + i) );

        int nhalodir[3] =  {
            d[0] != 0 ? 1 : XS,
            d[1] != 0 ? 1 : YS,
            d[2] != 0 ? 1 : ZS
        };

        int nhalocells = nhalodir[0] * nhalodir[1] * nhalodir[2];
        int estimate = numberdensity * safety_factor * nhalocells;
        CC(cudaMalloc(&packbuffers[i].scattered_indices, sizeof(int) * estimate));

        if (i && estimate) {
            CC(cudaHostAlloc(&pinnedhost_sendbufs[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
            CC(cudaHostGetDevicePointer(&packbuffers[i].buffer, pinnedhost_sendbufs[i], 0));

            CC(cudaHostAlloc(&pinnedhost_recvbufs[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
            CC(cudaHostGetDevicePointer(&unpackbuffers[i].buffer, pinnedhost_recvbufs[i], 0));
        } else {
            CC(cudaMalloc(&packbuffers[i].buffer, sizeof(float) * 6 * estimate));
            unpackbuffers[i].buffer = packbuffers[i].buffer;
            pinnedhost_sendbufs[i] = NULL;
            pinnedhost_recvbufs[i] = NULL;
        }
        packbuffers[i].capacity = estimate;
        unpackbuffers[i].capacity = estimate;
        default_message_sizes[i] = estimate;
    }

    k_odstr::texAllParticles.channelDesc = cudaCreateChannelDesc<float>();
    k_odstr::texAllParticles.filterMode = cudaFilterModePoint;
    k_odstr::texAllParticles.mipmapFilterMode = cudaFilterModePoint;
    k_odstr::texAllParticles.normalized = 0;

    k_odstr::texAllParticlesFloat2.channelDesc = cudaCreateChannelDesc<float2>();
    k_odstr::texAllParticlesFloat2.filterMode = cudaFilterModePoint;
    k_odstr::texAllParticlesFloat2.mipmapFilterMode = cudaFilterModePoint;
    k_odstr::texAllParticlesFloat2.normalized = 0;

    CC(cudaEventCreate(&evpacking, cudaEventDisableTiming));
    CC(cudaEventCreate(&evsizes, cudaEventDisableTiming));
    CC(cudaFuncSetCacheConfig(k_odstr::gather_particles, cudaFuncCachePreferL1));
}

void _post_recv() {
    for(int i = 1, c = 0; i < 27; ++i)
    if (default_message_sizes[i])
	MC(MPI_Irecv(recv_sizes + i, 1, MPI_INTEGER, neighbor_ranks[i],
                 BT_C_ODSTR + recv_tags[i], cart, recvcountreq + c++));
    else
	recv_sizes[i] = 0;

    for(int i = 1, c = 0; i < 27; ++i)
    if (default_message_sizes[i])
	MC( MPI_Irecv(pinnedhost_recvbufs[i], default_message_sizes[i] * 6, MPI_FLOAT,
                  neighbor_ranks[i], BT_P_ODSTR + recv_tags[i], cart, recvmsgreq + c++) );
}

void _adjust_send_buffers(int requested_capacities[27]) {
    for(int i = 0; i < 27; ++i) {
        if (requested_capacities[i] <= packbuffers[i].capacity)
        continue;

        int capacity = requested_capacities[i];

        CC(cudaFree(packbuffers[i].scattered_indices));
        CC(cudaMalloc(&packbuffers[i].scattered_indices, sizeof(int) * capacity));

        if (i) {
            CC(cudaFreeHost(pinnedhost_sendbufs[i]));

            CC(cudaHostAlloc(&pinnedhost_sendbufs[i], sizeof(float) * 6 * capacity, cudaHostAllocMapped));
            CC(cudaHostGetDevicePointer(&packbuffers[i].buffer, pinnedhost_sendbufs[i], 0));

            packbuffers[i].capacity = capacity;
        }
        else {
            CC(cudaFree(packbuffers[i].buffer));

            CC(cudaMalloc(&packbuffers[i].buffer, sizeof(float) * 6 * capacity));
            unpackbuffers[i].buffer = packbuffers[i].buffer;

            packbuffers[i].capacity = capacity;
            unpackbuffers[i].capacity = capacity;
        }
    }
}

bool _adjust_recv_buffers(int requested_capacities[27]) {
    bool haschanged = false;
    for(int i = 0; i < 27; ++i) {
        if (requested_capacities[i] <= unpackbuffers[i].capacity) continue;
        haschanged = true;
        int capacity = requested_capacities[i];
        if (i) {
            //preserve-resize policy
            float * old = pinnedhost_recvbufs[i];
            CC(cudaHostAlloc(&pinnedhost_recvbufs[i], sizeof(float) * 6 * capacity, cudaHostAllocMapped));
            CC(cudaHostGetDevicePointer(&unpackbuffers[i].buffer, pinnedhost_recvbufs[i], 0));
            CC(cudaMemcpy(pinnedhost_recvbufs[i], old, sizeof(float) * 6 * unpackbuffers[i].capacity,
                          H2H));
            CC(cudaFreeHost(old));
        }
        else {
            printf("_adjust_recv_buffers i==0 ooooooooooooooops %d , req %d!!\n", unpackbuffers[i].capacity, capacity);
            abort();
        }
        unpackbuffers[i].capacity = capacity;
    }
    return haschanged;
}

void pack(Particle * particles, int nparticles) {
    bool secondchance = false;
    if (firstcall) _post_recv();
    size_t textureoffset;
    if (nparticles)
    CC(cudaBindTexture(&textureoffset, &k_odstr::texAllParticles, particles,
                       &k_odstr::texAllParticles.channelDesc,
                       sizeof(float) * 6 * nparticles));

    if (nparticles)
    CC(cudaBindTexture(&textureoffset, &k_odstr::texAllParticlesFloat2, particles,
                       &k_odstr::texAllParticlesFloat2.channelDesc,
                       sizeof(float) * 6 * nparticles));

    k_odstr::ntexparticles = nparticles;
    k_odstr::texparticledata = (float2 *)particles;
 pack_attempt:
    CC(cudaMemcpyToSymbolAsync(k_odstr::pack_buffers, packbuffers,
                               sizeof(PackBuffer) * 27, 0, H2D));

    (*failure->D) = false;
    k_odstr::setup<<<1, 32>>>();

    if (nparticles)
    k_odstr::scatter_halo_indices_pack<<<k_cnf(nparticles)>>>(nparticles);

    k_odstr::tiny_scan<<<1, 32>>>
        (nparticles, packbuffers[0].capacity, packsizes->DP, failure->DP);

    CC(cudaEventRecord(evsizes));
    if (nparticles)
    k_odstr::pack<<<k_cnf(3 * nparticles)>>>
        (nparticles, nparticles * 3);

    CC(cudaEventRecord(evpacking));

    CC(cudaEventSynchronize(evsizes));

    if (*failure->D) {
        //wait for packing to finish
        CC(cudaEventSynchronize(evpacking));

        printf("pack RANK %d ...FAILED! Recovering now...\n", m::rank);

        _adjust_send_buffers(packsizes->D);

        if (m::rank == 0)
        for(int i = 0; i < 27; ++i)
        printf("ASD: %d\n", packsizes->D[i]);

        if (secondchance) {
            printf("...non siamo qui a far la ceretta allo yeti.\n");
            abort();
        }
        if (!secondchance) secondchance = true;
        goto pack_attempt;
    }

}

void send() {
    if (!firstcall) _waitall(sendcountreq, nactiveneighbors);
    for(int i = 0; i < 27; ++i) send_sizes[i] = packsizes->D[i];
    nbulk = recv_sizes[0] = send_sizes[0];
    {
        int c = 0;
        for(int i = 1; i < 27; ++i)
        if (default_message_sizes[i])
        MC(MPI_Isend(send_sizes + i, 1, MPI_INTEGER, neighbor_ranks[i],
                     BT_C_ODSTR + i, cart, sendcountreq + c++));
    }
    CC(cudaEventSynchronize(evpacking));
    if (!firstcall) _waitall(sendmsgreq, nsendmsgreq);

    nsendmsgreq = 0;
    for(int i = 1; i < 27; ++i)
    if (default_message_sizes[i]) {
        MC(MPI_Isend(pinnedhost_sendbufs[i], default_message_sizes[i] * 6, MPI_FLOAT,
                     neighbor_ranks[i], BT_P_ODSTR + i,
                     cart, sendmsgreq + nsendmsgreq) );

        ++nsendmsgreq;
    }

    for(int i = 1; i < 27; ++i)
    if (default_message_sizes[i] && send_sizes[i] > default_message_sizes[i]) {
        int count = send_sizes[i] - default_message_sizes[i];

        MC( MPI_Isend(pinnedhost_sendbufs[i] + default_message_sizes[i] * 6, count * 6, MPI_FLOAT,
                      neighbor_ranks[i], BT_P2_ODSTR + i, cart, sendmsgreq + nsendmsgreq) );
        ++nsendmsgreq;
    }
}

void bulk(int nparticles, int * cellstarts, int * cellcounts) {
    CC(cudaMemsetAsync(cellcounts, 0, sizeof(int) * XS * YS * ZS));

    subindices->resize(nparticles);

    if (nparticles)
    k_common::subindex_local<true><<<k_cnf(nparticles)>>>
        (nparticles, k_odstr::texparticledata, cellcounts, subindices->D);


}

int recv_count() {


    _waitall(recvcountreq, nactiveneighbors);

    {
        static int usize[27], ustart[28], ustart_padded[28];

        usize[0] = 0;
        for(int i = 1; i < 27; ++i)
        usize[i] = recv_sizes[i] * (default_message_sizes[i] > 0);

        ustart[0] = 0;
        for(int i = 1; i < 28; ++i)
        ustart[i] = ustart[i - 1] + usize[i - 1];

        nexpected = nbulk + ustart[27];
        nhalo = ustart[27];

        ustart_padded[0] = 0;
        for(int i = 1; i < 28; ++i)
        ustart_padded[i] = ustart_padded[i - 1] + 32 * ((usize[i - 1] + 31) / 32);

        nhalo_padded = ustart_padded[27];

        CC(cudaMemcpyToSymbolAsync(k_odstr::unpack_start, ustart,
                                   sizeof(int) * 28, 0, H2D));

        CC(cudaMemcpyToSymbolAsync(k_odstr::unpack_start_padded, ustart_padded,
                                   sizeof(int) * 28, 0, H2D));
    }

    {
        remote_particles->resize(nhalo);
        subindices_remote->resize(nhalo);
        scattered_indices->resize(nexpected);
    }

    firstcall = false;
    return nexpected;
}

void recv_unpack(Particle * particles, float4 * xyzouvwo, ushort4 * xyzo_half, int nparticles,
                 int * cellstarts, int * cellcounts) {
    _waitall(recvmsgreq, nactiveneighbors);

    bool haschanged = true;
    _adjust_recv_buffers(recv_sizes);

    if (haschanged)
    CC(cudaMemcpyToSymbolAsync(k_odstr::unpack_buffers, unpackbuffers,
                               sizeof(UnpackBuffer) * 27, 0, H2D));

    for(int i = 1; i < 27; ++i)
    if (default_message_sizes[i] && recv_sizes[i] > default_message_sizes[i]) {
        int count = recv_sizes[i] - default_message_sizes[i];

        MPI_Status status;
        MC( MPI_Recv(pinnedhost_recvbufs[i] + default_message_sizes[i] * 6, count * 6, MPI_FLOAT,
                     neighbor_ranks[i], BT_P2_ODSTR + recv_tags[i], cart, &status) );
    }


#ifndef NDEBUG
    CC(cudaMemset(remote_particles->D, 0xff, sizeof(Particle) * remote_particles->S));
#endif

    if (nhalo)
    k_odstr::subindex_remote<<<k_cnf(nhalo_padded)>>>
        (nhalo_padded, nhalo, cellcounts, (float2 *)remote_particles->D, subindices_remote->D);

    if (compressed_cellcounts->S)
    k_common::compress_counts<<<k_cnf(compressed_cellcounts->S)>>>
        (compressed_cellcounts->S, (int4 *)cellcounts, (uchar4 *)compressed_cellcounts->D);

    k_scan::scan(compressed_cellcounts->D, compressed_cellcounts->S, (uint *)cellstarts);

#ifndef NDEBUG
    CC(cudaMemset(scattered_indices->D, 0xff, sizeof(int) * scattered_indices->S));
#endif

    if (subindices->S)
    k_odstr::scatter_indices<<<k_cnf(subindices->S)>>>
        (false, subindices->D, subindices->S, cellstarts, scattered_indices->D, scattered_indices->S);

    if (nhalo)
    k_odstr::scatter_indices<<<k_cnf(nhalo)>>>
        (true, subindices_remote->D, nhalo, cellstarts, scattered_indices->D, scattered_indices->S);

    if (nparticles)
    k_odstr::gather_particles<<<k_cnf(nparticles)>>>
        (scattered_indices->D, (float2 *)remote_particles->D, nhalo,
         k_odstr::ntexparticles, nparticles, (float2 *)particles, xyzouvwo, xyzo_half);

    _post_recv();
}

void _cancel_recv() {
    if (!firstcall) {
        _waitall(sendcountreq, nactiveneighbors);
        _waitall(sendmsgreq, nsendmsgreq);

        for(int i = 0; i < nactiveneighbors; ++i)
        MC( MPI_Cancel(recvcountreq + i) );

        for(int i = 0; i < nactiveneighbors; ++i)
        MC( MPI_Cancel(recvmsgreq + i) );

        firstcall = true;
    }
}


void redist_part_close() {
    CC(cudaEventDestroy(evpacking));
    CC(cudaEventDestroy(evsizes));

    _cancel_recv();

    for(int i = 0; i < 27; ++i) {
        CC(cudaFree(packbuffers[i].scattered_indices));
        if (i) CC(cudaFreeHost(packbuffers[i].buffer));
        else   CC(cudaFree(packbuffers[i].buffer));
    }

    delete failure;
    delete packsizes;
    delete compressed_cellcounts;
    delete remote_particles;
    delete subindices_remote;
    delete subindices;
    delete scattered_indices;
}
}
