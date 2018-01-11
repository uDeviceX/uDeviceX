void drig_post_recv(Comm *c, Unpack *u) {
    UC(post_recv(&u->hipp, &c->ipp));
    UC(post_recv(&u->hss, &c->ss));
}

void drig_post_send(Pack *p, Comm *c) {
    UC(post_send(&p->hipp, &c->ipp));
    UC(post_send(&p->hss, &c->ss));
}

void drig_wait_recv(Comm *c, Unpack *u) {
    UC(wait_recv(&c->ipp, &u->hipp));
    UC(wait_recv(&c->ss, &u->hss));
}

void drig_wait_send(Comm *c) {
    wait_send(&c->ipp);
    wait_send(&c->ss);
}
