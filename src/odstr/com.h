namespace odstr {

void post_recv_pp(TicketD *t) {
    sub::post_recv(t->rank, t->btc, t->btp, /**/ t->recv_sz_req, t->recv_pp_req, &t->r);
}

void send_pp(TicketD *t) {
    if (!t->first) {
        sub::waitall_s(t->send_sz_req);
        sub::waitall_s(t->send_pp_req);
    }
    t->first = false;
    t->nbulk = sub::count_sz( /**/ &t->s);
    sub::send_sz(t->rank, t->btc, &t->s, /**/ t->send_sz_req);
    sub::send_pp(t->rank, t->btp, /**/ &t->s, t->send_pp_req);
}

void recv_pp(TicketD *t) {
    sub::waitall_r(t->recv_sz_req);
    sub::count(/**/ &t->r, &t->nhalo);
    sub::waitall_r(t->recv_pp_req);
}
}
