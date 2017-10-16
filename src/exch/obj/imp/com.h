void post_recv(Comm *c, Unpack *u) {
    post_recv(&u->hpp, &c->pp);
}

void post_send(Pack *p, Comm *c) {
    post_send(&p->hpp, &c->pp);
}

void wait_recv(Comm *c, Unpack *u) {
    UC(wait_recv(&c->pp, &u->hpp));
}

void wait_send(Comm *c) {
    wait_send(&c->pp);
}

void post_recv_ff(Comm *c, UnpackF *u) {
    post_recv(&u->hff, &c->ff);
}

void post_send_ff(PackF *p, Comm *c) {
    post_send(&p->hff, &c->ff);
}

void wait_recv_ff(Comm *c, UnpackF *u) {
    UC(wait_recv(&c->ff, &u->hff));
}

void wait_send_ff(Comm *c) {
    wait_send(&c->ff);
}
