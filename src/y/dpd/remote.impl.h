namespace dpd {
void upd_bag(SendHalo* sendhalos[]) {
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

void post_expected_recv(MPI_Comm cart, RecvHalo* recvhalos[]) {
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

void wait_send() {
  MPI_Status statuses[26 * 2];
  MC(l::m::Waitall(26, sendcellsreq, statuses));
  MC(l::m::Waitall(nsendreq, sendreq, statuses));
  MC(l::m::Waitall(26, sendcountreq, statuses));
}

void cancel_recv() {
  int i;
  for (i = 0; i < 26; ++i) MC(MPI_Cancel(recvreq + i));
  for (i = 0; i < 26; ++i) MC(MPI_Cancel(recvcellsreq + i));
  for (i = 0; i < 26; ++i) MC(MPI_Cancel(recvcountreq + i));
}

void fin(bool first) {
    CC(cudaFreeHost(required_send_bag_size));
    if (!first) {
      wait_send();
      cancel_recv();
    }
    CC(cudaEventDestroy(evfillall));
    CC(cudaEventDestroy(evdownloaded));
}
}
