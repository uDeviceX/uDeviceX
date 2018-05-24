void eobj_post_recv(EObjComm *c, EObjUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void eobj_post_send(EObjPack *p, EObjComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void eobj_wait_recv(EObjComm *c, EObjUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void eobj_wait_send(EObjComm *c) {
    UC(comm_wait_send(c->comm));
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
