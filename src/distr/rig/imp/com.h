void drig_post_recv(DRigComm *c, DRigUnpack *u) {
    UC(post_recv(&u->hipp, c->ipp));
    UC(post_recv(&u->hss, c->ss));
}

void drig_post_send(DRigPack *p, DRigComm *c) {
    UC(post_send(&p->hipp, c->ipp));
    UC(post_send(&p->hss, c->ss));
}

void drig_wait_recv(DRigComm *c, DRigUnpack *u) {
    UC(wait_recv(c->ipp, &u->hipp));
    UC(wait_recv(c->ss, &u->hss));
}

void drig_wait_send(DRigComm *c) {
    wait_send(c->ipp);
    wait_send(c->ss);
}
