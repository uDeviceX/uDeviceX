void eobj_post_recv(Comm *c, Unpack *u) {
    UC(post_recv(&u->hpp, &c->pp));
}

void eobj_post_send(Pack *p, Comm *c) {
    UC(post_send(&p->hpp, &c->pp));
}

void eobj_wait_recv(Comm *c, Unpack *u) {
    UC(wait_recv(&c->pp, &u->hpp));
}

void eobj_wait_send(Comm *c) {
    wait_send(&c->pp);
}

void eobj_post_recv_ff(Comm *c, UnpackF *u) {
    UC(post_recv(&u->hff, &c->ff));
}

void eobj_post_send_ff(PackF *p, Comm *c) {
    UC(post_send(&p->hff, &c->ff));
}

void eobj_wait_recv_ff(Comm *c, UnpackF *u) {
    UC(wait_recv(&c->ff, &u->hff));
}

void eobj_wait_send_ff(Comm *c) {
    UC(wait_send(&c->ff));
}
