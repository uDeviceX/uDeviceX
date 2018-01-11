void eobj_post_recv(EObjComm *c, EObjUnpack *u) {
    UC(post_recv(&u->hpp, &c->pp));
}

void eobj_post_send(EObjPack *p, EObjComm *c) {
    UC(post_send(&p->hpp, &c->pp));
}

void eobj_wait_recv(EObjComm *c, EObjUnpack *u) {
    UC(wait_recv(&c->pp, &u->hpp));
}

void eobj_wait_send(EObjComm *c) {
    wait_send(&c->pp);
}

void eobj_post_recv_ff(EObjComm *c, EObjUnpackF *u) {
    UC(post_recv(&u->hff, &c->ff));
}

void eobj_post_send_ff(EObjPackF *p, EObjComm *c) {
    UC(post_send(&p->hff, &c->ff));
}

void eobj_wait_recv_ff(EObjComm *c, EObjUnpackF *u) {
    UC(wait_recv(&c->ff, &u->hff));
}

void eobj_wait_send_ff(EObjComm *c) {
    UC(wait_send(&c->ff));
}
