namespace dpd {

bool check_size(SendHalo* sendhalos[]) {
  bool succeeded = true;
  for (int i = 0; i < 26; ++i) {
    int nrequired = required_send_bag_size_host[i];
    bool failed_entry = nrequired > sendhalos[i]->dbuf->C;

    if (failed_entry) {
      sendhalos[i]->dbuf->resize(nrequired);
      sendhalos[i]->scattered_entries->resize(nrequired);
      succeeded = false;
    }
  }
  return succeeded;
}
  
void post(MPI_Comm cart, Particle *pp, SendHalo* sendhalos[], int n) {
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
      upd_bag(sendhalos);
      pack(pp, n);
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
}
}
