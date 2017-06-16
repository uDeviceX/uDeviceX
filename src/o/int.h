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
