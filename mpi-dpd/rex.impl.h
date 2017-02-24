namespace rex {
  void _not_nan(float*, int) {};
  void bind_solutes(std::vector<ParticlesWrap> wsolutes_) {wsolutes = wsolutes_;}
  void _adjust_packbuffers() {
    int s = 0;
    for (int i = 0; i < 26; ++i) s += 32 * ((local[i]->capacity() + 31) / 32);
    packbuf->resize(s);
    host_packbuf->resize(s);
  }

  void init(MPI_Comm _cartcomm) {
    iterationcount = -1;
    packstotalstart = new DeviceBuffer<int>(27);
    host_packstotalstart = new PinnedHostBuffer<int>(27);
    host_packstotalcount = new PinnedHostBuffer<int>(26);

    packscount = new DeviceBuffer<int>;
    packsstart = new DeviceBuffer<int>;
    packsoffset = new DeviceBuffer<int>;

    packbuf = new DeviceBuffer<Particle>;
    host_packbuf = new PinnedHostBuffer<Particle>;

    for (int i = 0; i < SE_HALO_SIZE; i++) local[i] = new LocalHalo;
    for (int i = 0; i < SE_HALO_SIZE; i++) remote[i] = new RemoteHalo;

    MC(MPI_Comm_dup(_cartcomm, &cartcomm));
    MC(MPI_Comm_size(cartcomm, &nranks));
    MC(MPI_Cart_get(cartcomm, 3, dims, periods, coords));
    MC(MPI_Comm_rank(cartcomm, &myrank));

    for (int i = 0; i < 26; ++i) {
      int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

      recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

      int coordsneighbor[3];
      for (int c = 0; c < 3; ++c) coordsneighbor[c] = coords[c] + d[c];

      MC(MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i));

      int estimate = 1;
      remote[i]->preserve_resize(estimate);
      local[i]->resize(estimate);
      local[i]->update();

      CC(cudaMemcpyToSymbol(k_rex::ccapacities,
			    &local[i]->scattered_indices->C, sizeof(int),
			    sizeof(int) * i, cudaMemcpyHostToDevice));
      CC(cudaMemcpyToSymbol(k_rex::scattered_indices,
			    &local[i]->scattered_indices->D, sizeof(int *),
			    sizeof(int *) * i, cudaMemcpyHostToDevice));
    }

    _adjust_packbuffers();

    CC(cudaEventCreateWithFlags(&evPpacked,
				cudaEventDisableTiming | cudaEventBlockingSync));
    CC(cudaEventCreateWithFlags(&evAcomputed,
				cudaEventDisableTiming | cudaEventBlockingSync));
  }

  void _wait(std::vector<MPI_Request> &v) {
    MPI_Status statuses[v.size()];
    if (v.size()) MC(MPI_Waitall(v.size(), &v.front(), statuses));
    v.clear();
  }

  void _postrecvC() {
    for (int i = 0; i < 26; ++i) {
      MPI_Request reqC;
      MC(MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i],
		   TAGBASE_C + recv_tags[i], cartcomm, &reqC));
      reqrecvC.push_back(reqC);
    }
  }

  void _postrecvP() {
    for (int i = 0; i < 26; ++i) {
      MPI_Request reqP;
      remote[i]->pmessage.resize(remote[i]->expected());
      MC(MPI_Irecv(&remote[i]->pmessage.front(), remote[i]->expected() * 6,
		   MPI_FLOAT, dstranks[i], TAGBASE_P + recv_tags[i],
		   cartcomm, &reqP));
      reqrecvP.push_back(reqP);
    }
  }

  void _postrecvA() {
    for (int i = 0; i < 26; ++i) {
      MPI_Request reqA;

      MC(MPI_Irecv(local[i]->result->D, local[i]->result->S * 3,
		   MPI_FLOAT, dstranks[i], TAGBASE_A + recv_tags[i],
		   cartcomm, &reqA));
      reqrecvA.push_back(reqA);
    }
  }

  void _pack_attempt() {


    if (packscount->S)
      CC(cudaMemsetAsync(packscount->D, 0, sizeof(int) * packscount->S));

    if (packsoffset->S)
      CC(cudaMemsetAsync(packsoffset->D, 0, sizeof(int) * packsoffset->S));

    if (packsstart->S)
      CC(cudaMemsetAsync(packsstart->D, 0, sizeof(int) * packsstart->S));

    k_rex::init<<<1, 1, 0>>>();

    for (int i = 0; i < wsolutes.size(); ++i) {
      ParticlesWrap it = wsolutes[i];

      if (it.n) {
	CC(cudaMemcpyToSymbolAsync(k_rex::coffsets, packsoffset->D + 26 * i,
				   sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice));

	k_rex::scatter_indices<<<(it.n + 127) / 128, 128, 0>>>(
									   (float2 *)it.p, it.n, packscount->D + i * 26);
      }

      k_rex::tiny_scan<<<1, 32, 0>>>(
						 packscount->D + i * 26, packsoffset->D + 26 * i,
						 packsoffset->D + 26 * (i + 1), packsstart->D + i * 27);


    }

    CC(cudaMemcpyAsync(host_packstotalcount->D,
		       packsoffset->D + 26 * wsolutes.size(), sizeof(int) * 26,
		       cudaMemcpyDeviceToHost));

    k_rex::tiny_scan<<<1, 32, 0>>>(
					       packsoffset->D + 26 * wsolutes.size(), NULL, NULL, packstotalstart->D);

    CC(cudaMemcpyAsync(host_packstotalstart->D, packstotalstart->D,
		       sizeof(int) * 27, cudaMemcpyDeviceToHost));

    CC(cudaMemcpyToSymbolAsync(k_rex::cbases, packstotalstart->D,
			       sizeof(int) * 27, 0, cudaMemcpyDeviceToDevice));


    for (int i = 0; i < wsolutes.size(); ++i) {
      ParticlesWrap it = wsolutes[i];

      if (it.n) {
	CC(cudaMemcpyToSymbolAsync(k_rex::coffsets, packsoffset->D + 26 * i,
				   sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice));
	CC(cudaMemcpyToSymbolAsync(k_rex::ccounts, packscount->D + 26 * i,
				   sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice));
	CC(cudaMemcpyToSymbolAsync(k_rex::cpaddedstarts,
				   packsstart->D + 27 * i, sizeof(int) * 27, 0,
				   cudaMemcpyDeviceToDevice));

	k_rex::pack<<<14 * 16, 128, 0>>>
	  ((float2 *)it.p, it.n, (float2 *)packbuf->D, packbuf->C, i);
      }
    }

    CC(cudaEventRecord(evPpacked));


  }

  void pack_p() {
    if (wsolutes.size() == 0) return;

    ++iterationcount;

    packscount->resize(26 * wsolutes.size());
    packsoffset->resize(26 * (wsolutes.size() + 1));
    packsstart->resize(27 * wsolutes.size());

    _pack_attempt();
  }

  void post_p() {
    if (wsolutes.size() == 0) return;



    // consolidate the packing
    {
      CC(cudaEventSynchronize(evPpacked));

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
				   cudaMemcpyHostToDevice));

	int *newindices[26];
	for (int i = 0; i < 26; ++i) newindices[i] = local[i]->scattered_indices->D;

	CC(cudaMemcpyToSymbolAsync(k_rex::scattered_indices, newindices,
				   sizeof(newindices), 0, cudaMemcpyHostToDevice));

	_adjust_packbuffers();

	_pack_attempt();

	CC(cudaDeviceSynchronize()); /* was CC(cudaStreamSynchronize(stream)); */
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
			   cudaMemcpyDeviceToHost));
      }
      CC(cudaDeviceSynchronize()); /* was CC(cudaStreamSynchronize(downloadstream)); */
    }

    // post the sending of the packs
    {
      reqsendC.resize(26);

      for (int i = 0; i < 26; ++i)
	MC(MPI_Isend(send_counts + i, 1, MPI_INTEGER, dstranks[i],
		     TAGBASE_C + i, cartcomm, &reqsendC[i]));

      for (int i = 0; i < 26; ++i) {
	int start = host_packstotalstart->D[i];
	int count = send_counts[i];
	int expected = local[i]->expected();

	MPI_Request reqP;

	_not_nan((float *)(host_packbuf->D + start), count * 6);

	MC(MPI_Isend(host_packbuf->D + start, expected * 6, MPI_FLOAT,
		     dstranks[i], TAGBASE_P + i, cartcomm, &reqP));
	reqsendP.push_back(reqP);

	if (count > expected) {
	  MPI_Request reqP2;
	  MC(MPI_Isend(host_packbuf->D + start + expected,
		       (count - expected) * 6, MPI_FLOAT, dstranks[i],
		       TAGBASE_P2 + i, cartcomm, &reqP2));

	  reqsendP.push_back(reqP2);
	}
      }
    }
  }

  void recv_p() {
    if (wsolutes.size() == 0) return;

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
		    TAGBASE_P2 + recv_tags[i], cartcomm, &status));

      memcpy(remote[i]->hstate.D, &remote[i]->pmessage.front(),
	     sizeof(Particle) * count);
      _not_nan((float *)remote[i]->hstate.D, count * 6);
    }

    _postrecvC();

    for (int i = 0; i < 26; ++i)
      CC(cudaMemcpyAsync(remote[i]->dstate.D, remote[i]->hstate.D,
			 sizeof(Particle) * remote[i]->hstate.S,
			 cudaMemcpyHostToDevice));
  }

  void halo() {
    if (wsolutes.size() == 0) return;

    if (iterationcount) _wait(reqsendA);

    ParticlesWrap halos[26];

    for (int i = 0; i < 26; ++i)
      halos[i] = ParticlesWrap(remote[i]->dstate.D, remote[i]->dstate.S,
			       remote[i]->result.DP);

    CC(cudaDeviceSynchronize()); /* was CC(cudaStreamSynchronize(uploadstream)); */

    /** here was visitor  **/
    fsi::halo(halos);
    if (contactforces) cnt::halo(halos);
    /***********************/



    CC(cudaEventRecord(evAcomputed));

    for (int i = 0; i < 26; ++i) local[i]->update();

    _postrecvP();
  }

  void post_a() {
    if (wsolutes.size() == 0) return;

    CC(cudaEventSynchronize(evAcomputed));

    reqsendA.resize(26);
    for (int i = 0; i < 26; ++i)
      MC(MPI_Isend(remote[i]->result.D, remote[i]->result.S * 3,
		   MPI_FLOAT, dstranks[i], TAGBASE_A + i, cartcomm,
		   &reqsendA[i]));
  }

  void recv_a() {


    if (wsolutes.size() == 0) return;

    {
      float *recvbags[26];

      for (int i = 0; i < 26; ++i) recvbags[i] = (float *)local[i]->result->DP;

      CC(cudaMemcpyToSymbolAsync(k_rex::recvbags, recvbags, sizeof(recvbags),
				 0, cudaMemcpyHostToDevice));
    }

    _wait(reqrecvA);

    for (int i = 0; i < wsolutes.size(); ++i) {
      ParticlesWrap it = wsolutes[i];

      if (it.n) {
	CC(cudaMemcpyToSymbolAsync(k_rex::cpaddedstarts,
				   packsstart->D + 27 * i, sizeof(int) * 27, 0,
				   cudaMemcpyDeviceToDevice));
	CC(cudaMemcpyToSymbolAsync(k_rex::ccounts, packscount->D + 26 * i,
				   sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice));
	CC(cudaMemcpyToSymbolAsync(k_rex::coffsets, packsoffset->D + 26 * i,
				   sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice));

	k_rex::unpack<<<16 * 14, 128, 0>>>((float *)it.a, it.n);
      }

    }
  }

  void close() {
    MC(MPI_Comm_free(&cartcomm));

    CC(cudaEventDestroy(evPpacked));
    CC(cudaEventDestroy(evAcomputed));

    delete packstotalstart;
    delete host_packstotalstart;
    delete host_packstotalcount;

    delete packscount;
    delete packsstart;
    delete packsoffset;
    delete packbuf;
    delete host_packbuf;

    for (int i = 0; i < SE_HALO_SIZE; i++) delete local[i];
    for (int i = 0; i < SE_HALO_SIZE; i++) delete remote[i];
  }
}
