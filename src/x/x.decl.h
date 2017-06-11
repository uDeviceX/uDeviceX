/* refuges from sim.impl.h */
MPI_Comm s_cart;
uchar4 *s_subi_lo, *s_subi_re;
uint   *iidx;
bool first = true;
MPI_Request send_size_req[27], recv_size_req[27],
  /*     */ send_mesg_req[27], recv_mesg_req[27];
int s_rank[27];

Particle *s_pp_re; /* remote particles */
unsigned char *s_count_zip;
