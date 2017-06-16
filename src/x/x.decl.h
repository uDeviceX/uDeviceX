MPI_Comm cart;
uchar4 *subi_lo, *subi_re; /* subindices local and remote */
uint   *iidx;
bool first = true;
MPI_Request send_size_req[27], recv_size_req[27],
            send_mesg_req[27], recv_mesg_req[27];
int rank[27];

Particle *pp_re; /* remote particles */
unsigned char *count_zip; /* compress cell counter */
