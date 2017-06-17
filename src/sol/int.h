/* struct Quants {
  Particle *pp;
  int       n;
  Clist *cells;
  }; */

struct TicketZ { /* zip */
  float4  *zip0;
  ushort4 *zip1;
};

struct TicketD { /* distribution */
  MPI_Comm cart;
  int rank[27];
  MPI_Request send_size_req[27], recv_size_req[27];
  MPI_Request send_mesg_req[27], recv_mesg_req[27];
  bool first = true;
  /* Odstr odstr; */
};

struct Work {
  uchar4 *subi_lo, *subi_re; /* local remote subindices */
  uint   *iidx;              /* scatter indices */
  Particle *pp_re;           /* remote particles */
  unsigned char *count_zip;
  Particle *pp0;
};

void alloc_work(Work *w) {
  mpDeviceMalloc(&w->subi_lo);
  mpDeviceMalloc(&w->subi_re);
  mpDeviceMalloc(&w->iidx);
  mpDeviceMalloc(&w->pp_re);
  CC(cudaMalloc(&w->count_zip, sizeof(w->count_zip[0])*XS*YS*ZS));
  mpDeviceMalloc(&w->pp0);
}

void free_work(Work *w) {
  CC(cudaFree(w->subi_lo));
  CC(cudaFree(w->subi_re));
  CC(cudaFree(w->iidx));
  CC(cudaFree(w->pp_re));
  CC(cudaFree(w->count_zip));
  /* pp0 shell not be freed? */
}

void alloc_ticketD(TicketD *t) {
  l::m::Comm_dup(m::cart, &t->cart);
  odstr::ini(t->cart, t->rank); /* :TODO: odstr is only one */
  t->first = true;
}

void free_ticketD() {
  odstr::fin();
}

void alloc_ticketZ(/**/ TicketZ *t) {
  mpDeviceMalloc(&t->zip0);
  mpDeviceMalloc(&t->zip1);
}

void free_ticketZ(/**/ TicketZ *t) {
  float4  *zip0 = t->zip0;
  ushort4 *zip1 = t->zip1;
  cudaFree(zip0);
  cudaFree(zip1);
}

void create_ticketZ(Particle *pp, int n, /**/ TicketZ *t) {
  if (n == 0) return;
  float4  *zip0 = t->zip0;
  ushort4 *zip1 = t->zip1;
  assert(sizeof(Particle) == 6 * sizeof(float)); /* :TODO: implicit dependency */
  sub::zip<<<(n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>(zip0, zip1, (float*)pp, n);
}

void distr(Particle *pp, int *qn, Clist *cells, TicketD *td, TicketZ *tz, Work *w) {
  MPI_Comm cart = td->cart; /* can be a copy */
  int *rank = td->rank; /* arrays */
  int *send_size_req = td->send_size_req;
  int *recv_size_req = td->recv_size_req;
  MPI_Request *send_mesg_req = td->send_mesg_req;
  MPI_Request *recv_mesg_req = td->recv_mesg_req;
  bool *qfirst = &td->first; /* shoud be updated */

  float4  *zip0 = tz->zip0;
  ushort4 *zip1 = tz->zip1;

  uchar4 *subi_lo = w->subi_lo; /* arrays */
  uchar4 *subi_re = w->subi_re;
  uint   *iidx = w->iidx;
  Particle *pp_re = w->pp_re;
  unsigned char *count_zip = w->count_zip;
  Particle *pp0 = w->pp0;

  int n = *qn;
  bool first = *qfirst;
  
  int nbulk, nhalo_padded, nhalo;
  odstr::post_recv(cart, rank, /**/ recv_size_req, recv_mesg_req);
  if (n) {
    odstr::halo(pp, n);
    odstr::scan(n);
    odstr::pack(pp, n);
  }
  if (!first) {
    odstr::waitall(send_size_req);
    odstr::waitall(send_mesg_req);
  }
  first = false;
  nbulk = odstr::send_sz(cart, rank, send_size_req);
  odstr::send_msg(cart, rank, send_mesg_req);

  CC(cudaMemsetAsync(cells->count, 0, sizeof(int)*XS*YS*ZS));
  if (n)
    k_common::subindex_local<false><<<k_cnf(n)>>>
      (n, (float2*)pp, /*io*/ cells->count, /*o*/ subi_lo);

  odstr::waitall(recv_size_req);
  odstr::recv_count(&nhalo_padded, &nhalo);
  odstr::waitall(recv_mesg_req);
  if (nhalo)
    odstr::unpack
      (nhalo_padded, /*io*/ cells->count, /*o*/ subi_re, pp_re);

  k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>
    (XS*YS*ZS, (int4*)cells->count, /**/ (uchar4*)count_zip);
  l::scan::d::scan(count_zip, XS*YS*ZS, /**/ (uint*)cells->start);

  if (n)
    k_odstr::scatter<<<k_cnf(n)>>>
      (false, subi_lo,  n, cells->start, /**/ iidx);

  if (nhalo)
    k_odstr::scatter<<<k_cnf(nhalo)>>>
      (true, subi_re, nhalo, cells->start, /**/ iidx);

  n = nbulk + nhalo;
  if (n)
    k_odstr::gather<<<k_cnf(n)>>>
      ((float2*)pp, (float2*)pp_re, n, iidx,
       /**/ (float2*)pp0, zip0, zip1);

  *qn = n;
  *qfirst = first;
}
