struct TicketCom { /* communication ticket */
    int btc, btp;           /* basetags     */
    MPI_Comm cart;          /* communicator */
    sub::Reqs sreq, rreq;   /* requests     */
    int recv_tags[26], dstranks[26];
    bool first;
};

struct TicketS { /* send data */
    Particle *pp_hst[27]; /* particles on host */
    int counts[27];       /* number of meshes  */
};

struct TicketR { /* recv data */
    Particle *pp_hst[27]; /* particles on host */
    int counts[27];       /* number of meshes  */
};

void ini_ticketcom(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ TicketCom *t) {
    sub::ini_tcom(cart, /**/ &t->cart, t->dstranks, t->recv_tags);
    t->first = true;
    t->btc = get_tag(tg);
    t->btp = get_tag(tg);
}

void free_ticketcom(/**/ TicketCom *t) {
    sub::fin_tcom(t->first, /**/ &t->cart, &t->sreq, &t->rreq);
}

void alloc_ticketS(TicketS *t) {
    for (int i = 0; i < 27; ++i) t->pp_hst[i] = new Particle[MAX_PART_NUM];
}

void free_ticketS(TicketS *t) {
    for (int i = 0; i < 27; ++i) delete[] t->pp_hst[i];
}

void alloc_ticketR(TicketR *t) {
    for (int i = 0; i < 27; ++i) t->pp_hst[i] = new Particle[MAX_PART_NUM];
}

void free_ticketR(TicketR *t) {
    for (int i = 0; i < 27; ++i) delete[] t->pp_hst[i];
}

