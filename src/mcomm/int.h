struct TicketCom { /* communication ticket */
    int btc, btp;           /* basetags     */
    MPI_Comm cart;          /* communicator */
    sub::Reqs sreq, rreq;   /* requests     */
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

void ini_ticketcom(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ TicketCom *t) {
    sub::ini_tcom(cart, /**/ &t->cart, t->dstranks, t->recv_tags);
    t->first = true;
    t->btc = get_tag(tg);
    t->btp = get_tag(tg);
}

void free_ticketcom(/**/ TicketCom *t) {
    sub::fin_tcom(t->first, /**/ &t->cart, &t->sreq, &t->rreq);
}
