namespace bbhalo
{    
    MPI_Comm cart;
    MPI_Request sendcntreq[26];
    std::vector<MPI_Request> sendreq, recvreq, recvcntreq;

    std::vector<Solid> shalo[27], rhalo[27];
    
    int rnk_ne[27]; /* rank      of the neighbor */
    int ank_ne[27]; /* anti-rank of the neighbor */
    int recv_counts[27], send_counts[27];
}
