void drig_post_recv(DRigComm *c, DRigUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void drig_post_send(DRigPack *p, DRigComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void drig_wait_recv(DRigComm *c, DRigUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void drig_wait_send(DRigComm *c) {
    UC(comm_wait_send(c->comm));
}
