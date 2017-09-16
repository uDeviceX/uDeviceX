void post_recv(Comm *c, Unpack *u) {
    post_recv(&u->hpp, &c->pp);
    if (global_ids)    post_recv(&u->hii, &c->ii);
    if (multi_solvent) post_recv(&u->hcc, &c->cc);
}

void post_send(Pack *p, Comm *c) {
    post_send(&p->hpp, &c->pp);
    if (global_ids)    post_send(&p->hii, &c->ii);
    if (multi_solvent) post_send(&p->hcc, &c->cc);
}

void wait_recv(Comm *c, Unpack *u) {
    wait_recv(&c->pp, &u->hpp);
    if (global_ids)    wait_recv(&c->ii, &u->hii);
    if (multi_solvent) wait_recv(&c->cc, &u->hcc);
}

void wait_send(Comm *c) {
    wait_send(&c->pp);
    if (global_ids)    wait_send(&c->ii);
    if (multi_solvent) wait_send(&c->cc);
}
