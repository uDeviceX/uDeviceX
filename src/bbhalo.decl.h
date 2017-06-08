namespace bbhalo
{    
MPI_Comm cart;
MPI_Request sendcntreq[26];
std::vector<MPI_Request> ssendreq, srecvreq, psendreq, precvreq, recvcntreq;

std::vector<Solid> sshalo[27]; /* [s]olid [s]end halo buffer */
std::vector<Solid> srhalo[27]; /* [s]olid [r]ecv halo buffer */

std::vector<Particle> pshalo[27]; /* [p]articles [s]end halo buffer */
std::vector<Particle> prhalo[27]; /* [p]articles [r]ecv halo buffer */
    
int rnk_ne[27]; /* rank      of the neighbor */
int ank_ne[27]; /* anti-rank of the neighbor */
int recv_counts[27], send_counts[27];
}
