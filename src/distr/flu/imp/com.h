void dflu_post_recv(DFluComm *c, DFluUnpack *u) {
    UC(comm_post_recv(&u->hpp, c->pp));
    if (c->opt.ids)    UC(comm_post_recv(&u->hii, c->ii));
    if (c->opt.colors) UC(comm_post_recv(&u->hcc, c->cc));
}

void dflu_post_send(DFluPack *p, DFluComm *c) {
    UC(comm_post_send(&p->hpp, c->pp));
    if (c->opt.ids)    UC(comm_post_send(&p->hii, c->ii));
    if (c->opt.colors) UC(comm_post_send(&p->hcc, c->cc));
}

void dflu_wait_recv(DFluComm *c, DFluUnpack *u) {
    UC(comm_wait_recv(c->pp, &u->hpp));
    if (c->opt.ids)    UC(comm_wait_recv(c->ii, &u->hii));
    if (c->opt.colors) UC(comm_wait_recv(c->cc, &u->hcc));
}

void dflu_wait_send(DFluComm *c) {
    UC(comm_wait_send(c->pp));
    if (c->opt.ids)    UC(comm_wait_send(c->ii));
    if (c->opt.colors) UC(comm_wait_send(c->cc));
}
