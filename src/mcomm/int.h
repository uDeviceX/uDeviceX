struct TicketCom { /* communication ticket */
    /* basetags */
    int btc, btp;
    MPI_Comm cart;
    sub::Reqs sreq, rreq; 
    int recv_tags[26], recv_counts[26], dstranks[26];
    bool first;
};

struct TicketS { /* send data */
    Particle *pp[27];
    int counts[27];
};

struct TicketR { /* recv data */
    Particle *pp[27];
    int counts[27];
};

