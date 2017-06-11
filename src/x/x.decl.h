/* refuges from sim.impl.h */
MPI_Comm s_cart;
uchar4 *s_subi_lo, *s_subi_re;
uint   *s_iidx;
bool s_first = true;
MPI_Request s_send_size_req[27], s_recv_size_req[27],
  /*     */ s_send_mesg_req[27], s_recv_mesg_req[27];
int s_rank[27];

Particle *s_pp_re; /* remote particles */
unsigned char *s_count_zip;
