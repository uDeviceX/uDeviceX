struct Quants {
  Particle *pp;
  int       n;
  Clist *cells;
}

struct TicketZip {
  float4  *zip0;
  ushort4 *zip1;
}

struct TicketDistr {
  MPI_Comm cart;
  int rank[27];
  MPI_Request send_size_req[27], recv_size_req[27];
  MPI_Request send_mesg_req[27], recv_mesg_req[27];
  bool first = true;
  /* Odstr odstr; */
}

struct Work {
  uchar4 *subi_lo, *subi_re;
  uint   *iidx;
  Particle *pp_re;
  unsigned char *count_zip
  Particle *pp0;
}
