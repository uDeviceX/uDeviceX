void drbc_post_recv(Comm *c, Unpack *u) {
    UC(post_recv(&u->hpp, &c->pp));
    if (rbc_ids) UC(post_recv(&u->hii, &c->ii));
}

void drbc_post_send(Pack *p, Comm *c) {
    UC(post_send(&p->hpp, &c->pp));
    if (rbc_ids) UC(post_send(&p->hii, &c->ii));
}

void drbc_wait_recv(Comm *c, Unpack *u) {
    UC(wait_recv(&c->pp, &u->hpp));
    if (rbc_ids) UC(wait_recv(&c->ii, &u->hii));
}

void drbc_wait_send(Comm *c) {
    UC(wait_send(&c->pp));
    if (rbc_ids) UC(wait_send(&c->ii));
}
