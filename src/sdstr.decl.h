namespace sdstr
{
std::vector<Solid> srbuf[27], ssbuf[27]; /* send and recieve buffers: solid */
std::vector<Particle> prbuf[27], psbuf[27]; /* send and recieve buffers: mesh */
    
MPI_Comm cart; /* Cartesian communicator */
MPI_Request sendcntreq[26];
  
std::vector<MPI_Request> ssendreq, srecvreq, psendreq, precvreq, recvcntreq;
int rnk_ne[27]; /* rank      of the neighbor */
int ank_ne[27]; /* anti-rank of the neighbor */
int recv_counts[27], send_counts[27];

int nstay;
}
