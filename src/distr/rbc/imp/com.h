void drbc_post_recv(DRbcComm *c, DRbcUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void drbc_post_send(DRbcPack *p, DRbcComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void drbc_wait_recv(DRbcComm *c, DRbcUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void drbc_wait_send(DRbcComm *c) {
    UC(comm_wait_send(c->comm));
}
