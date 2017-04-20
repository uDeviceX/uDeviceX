namespace rdstr
{
    std::vector<Solid> rbuf[27], sbuf[27]; /* send and recieve buffers */
    
    MPI_Comm cart; /* Cartesian communicator */
    MPI_Request sendcntreq[26];
  
    std::vector<MPI_Request> sendreq, recvreq, recvcntreq;
    int rnk_ne[27]; /* rank      of the neighbor */
    int ank_ne[27]; /* anti-rank of the neighbor */
    int recv_counts[27], send_counts[27];

    int nstay;
}
