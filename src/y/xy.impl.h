namespace xy { /* temporary interface to dpd/x and dpd/y */
void ini0() {
  int i;
  for (i = 0; i < 26; i++) recvhalos[i] = new RecvHalo;
  for (i = 0; i < 26; i++) sendhalos[i] = new SendHalo;
}

void ini() {
  int i;
  ini0();
  dpd::ini(interrank_trunks, interrank_masks);
  for (i = 0; i < 26; ++i) init1_one(i, interrank_trunks, interrank_masks);
}

void fin0() {
  int i;
  for (i = 0; i < 26; i++) delete recvhalos[i];
  for (i = 0; i < 26; i++) delete sendhalos[i];
}

void fin() {
  dpd::fin(interrank_trunks);
  fin0();
}

void forces(flu::Quants *q, flu::TicketZ *tz, flu::TicketRND *trnd, /**/ Force *ff) {
  dpd::pack(q->pp, q->n, q->cells->start, q->cells->count);
  dpd::flocal(tz->zip0, tz->zip1, q->n,
	      q->cells->start, q->cells->count,
	      trnd->rnd,
	      /**/ ff);
  dpd::post(q->pp, q->n);
  dpd::recv();
  dpd::fremote(q->n, interrank_trunks, interrank_masks, /**/ ff);
}
}
