namespace xy { /* temporary interface to dpd/x and dpd/y */
void ini0() {
  int i;
  for (i = 0; i < 26; i++) recvhalos[i] = new dpd::RecvHalo;
  for (i = 0; i < 26; i++) sendhalos[i] = new dpd::SendHalo;
}

void ini() {
  int i;
  first = true;
  ini0();
  MC(l::m::Comm_dup(m::cart, &cart));
  dpd::ini(cart, sendhalos, recvhalos);
  for (i = 0; i < 26; ++i) dpd::init1_one(i, /**/ interrank_trunks, interrank_masks);
}

void fin0() {
  int i;
  for (i = 0; i < 26; i++) delete recvhalos[i];
  for (i = 0; i < 26; i++) delete sendhalos[i];
  for (i = 1; i < 26; i++) delete interrank_trunks[i];
  MC(l::m::Comm_free(&cart));
}

void fin() {
  dpd::fin(first);
  fin0();
  first = true;
}

void forces(flu::Quants *q, flu::TicketZ *tz, flu::TicketRND *trnd, /**/ Force *ff) {
  if (first) dpd::pack_first0(sendhalos);
  dpd::scan(q->cells->start, q->cells->count);
  if (first) dpd::post_expected_recv(cart, recvhalos); else dpd::wait_send();
  if (first) dpd::pack_first1(sendhalos);
  dpd::copycells();
  if (first) dpd::upd_bag(sendhalos);
  first = false;
  dpd::pack(q->pp, q->n);
  
  dpd::flocal(tz->zip0, tz->zip1, q->n, q->cells->start, q->cells->count, trnd->rnd,
	      /**/ ff);
  dpd::post(cart, q->pp, sendhalos, q->n);
  dpd::wait_recv();
  dpd::recv(cart, recvhalos);
  dpd::post_expected_recv(cart, recvhalos);
  dpd::fremote(q->n, sendhalos, recvhalos, interrank_trunks, interrank_masks, /**/ ff);
}
}
