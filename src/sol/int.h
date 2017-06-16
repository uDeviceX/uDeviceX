struct Quants {
  Particle *pp;
  int       n;
  Clist *cells;
};

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

void alloc_ticketZ(/**/ TicketZ t) {
  float4  *zip0 = t.zip0;
  ushort4 *zip1 = t.zip1;
  mpDeviceMalloc(&zip0);
  mpDeviceMalloc(&zip1);
}

void create_ticketZ(Quants q, /**/ TicketZ t) {
  Particle  *pp = q.pp;
  int         n = q.n;
  float4  *zip0 = t.zip0;
  ushort4 *zip1 = t.zip1;
  assert(sizeof(Particle)/sizeof(float) == 6); /* TODO: implicit dependency */
  sub::zip<<<(n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>(zip0, zip1, (float*)pp, n);
}
