void post_recv(Comm *c, Unpack *u) {
    post_recv(&u->hpp, &c->pp);
}

void post_send(Pack *p, Comm *c) {
    post_send(&p->hpp, &c->pp);
}

void wait_recv(Comm *c, Unpack *u) {
    wait_recv(&c->pp, &u->hpp);
}

void wait_send(Comm *c) {
    wait_send(&c->pp);
}
