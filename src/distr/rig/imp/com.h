void post_recv(Comm *c, Unpack *u) {
    OC(post_recv(&u->hipp, &c->ipp));
    OC(post_recv(&u->hss, &c->ss));
}

void post_send(Pack *p, Comm *c) {
    UC(post_send(&p->hipp, &c->ipp));
    UC(post_send(&p->hss, &c->ss));
}

void wait_recv(Comm *c, Unpack *u) {
    UC(wait_recv(&c->ipp, &u->hipp));
    UC(wait_recv(&c->ss, &u->hss));
}

void wait_send(Comm *c) {
    wait_send(&c->ipp);
    wait_send(&c->ss);
}
