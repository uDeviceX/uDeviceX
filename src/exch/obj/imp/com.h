void eobj_post_recv(EObjComm *c, EObjUnpack *u) {
    UC(comm_post_recv(&u->hpp, c->pp));
}

void eobj_post_send(EObjPack *p, EObjComm *c) {
    UC(comm_post_send(&p->hpp, c->pp));
}

void eobj_wait_recv(EObjComm *c, EObjUnpack *u) {
    UC(comm_wait_recv(c->pp, &u->hpp));
}

void eobj_wait_send(EObjComm *c) {
    UC(comm_wait_send(c->pp));
}

void eobj_post_recv_ff(EObjCommF *c, EObjUnpackF *u) {
    UC(comm_post_recv(&u->hff, c->comm));
}

void eobj_post_send_ff(EObjPackF *p, EObjCommF *c) {
    UC(comm_post_send(&p->hff, c->comm));
}

void eobj_wait_recv_ff(EObjCommF *c, EObjUnpackF *u) {
    UC(comm_wait_recv(c->comm, &u->hff));
}

void eobj_wait_send_ff(EObjCommF *c) {
    UC(comm_wait_send(c->comm));
}
