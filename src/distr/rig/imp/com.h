void post_recv(Comm *c, Unpack *u) {
    post_recv(&u->hipp, &c->ipp);
    post_recv(&u->hss, &c->ss);
}

void post_send(Pack *p, Comm *c) {
    post_send(&p->hipp, &c->ipp);
    post_send(&p->hss, &c->ss);
}

void wait_recv(Comm *c, Unpack *u) {
    wait_recv(&c->ipp, &u->hipp);
    wait_recv(&c->ss, &u->hss);
}

void wait_send(Comm *c) {
    wait_send(&c->ipp);
    wait_send(&c->ss);
}
