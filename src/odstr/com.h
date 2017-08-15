namespace odstr {

void post_recv_pp(TicketD *t) {
    sub::post_recv(t->cart, t->rank, t->btc, t->btp, /**/ t->recv_sz_req, t->recv_pp_req, &t->r);
}

void send_pp(TicketD *t) {
    if (!t->first) {
        sub::waitall(t->send_sz_req);
        sub::waitall(t->send_pp_req);
        t->first = false;
    }
    t->nbulk = sub::count_sz( /**/ &t->s);
    sub::send_sz(t->cart, t->rank, t->btc, &t->s, /**/ t->send_sz_req);
    sub::send_pp(t->cart, t->rank, t->btp, /**/ &t->s, t->send_pp_req);
}

void recv_pp(TicketD *t) {
    sub::waitall(t->recv_sz_req);
    sub::count(/**/ &t->r, &t->nhalo);
    sub::waitall(t->recv_pp_req);
}
}
