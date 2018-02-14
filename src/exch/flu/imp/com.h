void eflu_post_recv(EFluComm *c, EFluUnpack *u) {
    UC(comm_post_recv(&u->hpp, c->pp));
    UC(comm_post_recv(&u->hfss, c->fss));
    if (c->opt.colors)
        UC(comm_post_recv(&u->hcc, c->cc));
}

void eflu_post_send(EFluPack *p, EFluComm *c) {
    UC(comm_post_send(&p->hpp, c->pp));
    UC(comm_post_send(&p->hfss, c->fss));
    if (c->opt.colors)
        UC(comm_post_send(&p->hcc, c->cc));
}

void eflu_wait_recv(EFluComm *c, EFluUnpack *u) {
    UC(comm_wait_recv(c->pp, &u->hpp));
    UC(comm_wait_recv(c->fss, &u->hfss));
    if (c->opt.colors)
        UC(comm_wait_recv(c->cc, &u->hcc));
}

void eflu_wait_send(EFluComm *c) {
    UC(comm_wait_send(c->pp));
    UC(comm_wait_send(c->fss));
    if (c->opt.colors)
        UC(comm_wait_send(c->cc));
}

