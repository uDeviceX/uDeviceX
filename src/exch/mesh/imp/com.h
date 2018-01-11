void emesh_post_recv(Comm *c, Unpack *u) {
    UC(post_recv(&u->hpp, &c->pp));
}

void emesh_post_send(Pack *p, Comm *c) {
    UC(post_send(&p->hpp, &c->pp));
}

void emesh_wait_recv(Comm *c, Unpack *u) {
    UC(wait_recv(&c->pp, &u->hpp));
}

void emesh_wait_send(Comm *c) {
    UC(wait_send(&c->pp));
}


/* momentum */

void emesh_post_recv(CommM *c, UnpackM *u) {
    UC(post_recv(&u->hmm, &c->mm));
    UC(post_recv(&u->hii, &c->ii));
}

void emesh_post_send(PackM *p, CommM *c) {
    UC(post_send(&p->hmm, &c->mm));
    UC(post_send(&p->hii, &c->ii));
}

void emesh_wait_recv(CommM *c, UnpackM *u) {
    UC(wait_recv(&c->mm, &u->hmm));
    UC(wait_recv(&c->ii, &u->hii));
}

void emesh_wait_send(CommM *c) {
    UC(wait_send(&c->mm));
    UC(wait_send(&c->ii));
}
