void dflu_post_recv(DFluComm *c, DFluUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void dflu_post_send(DFluPack *p, DFluComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void dflu_wait_recv(DFluComm *c, DFluUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void dflu_wait_send(DFluComm *c) {
    UC(comm_wait_send(c->comm));
}
