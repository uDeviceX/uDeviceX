void post_recv(Comm *c, Unpack *u) {
    post_recv(&u->hpp, &c->pp);
    if (rbc_ids) post_recv(&u->hii, &c->ii);
}

void post_send(Pack *p, Comm *c) {
    UC(post_send(&p->hpp, &c->pp));
    if (rbc_ids) UC(post_send(&p->hii, &c->ii));
}

void wait_recv(Comm *c, Unpack *u) {
    UC(wait_recv(&c->pp, &u->hpp));
    if (rbc_ids) UC(wait_recv(&c->ii, &u->hii));
}

void wait_send(Comm *c) {
    wait_send(&c->pp);
    if (rbc_ids) wait_send(&c->ii);
}
