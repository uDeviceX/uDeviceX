void drig_post_recv(DRigComm *c, DRigUnpack *u) {
    UC(comm_post_recv(&u->hipp, c->ipp));
    UC(comm_post_recv(&u->hss, c->ss));
}

void drig_post_send(DRigPack *p, DRigComm *c) {
    UC(comm_post_send(&p->hipp, c->ipp));
    UC(comm_post_send(&p->hss, c->ss));
}

void drig_wait_recv(DRigComm *c, DRigUnpack *u) {
    UC(comm_wait_recv(c->ipp, &u->hipp));
    UC(comm_wait_recv(c->ss, &u->hss));
}

void drig_wait_send(DRigComm *c) {
    UC(comm_wait_send(c->ipp));
    UC(comm_wait_send(c->ss));
}
