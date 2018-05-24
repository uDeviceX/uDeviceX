void eflu_post_recv(EFluComm *c, EFluUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void eflu_post_send(EFluPack *p, EFluComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void eflu_wait_recv(EFluComm *c, EFluUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void eflu_wait_send(EFluComm *c) {
    UC(comm_wait_send(c->comm));
}

