void post_recv(Comm *c, Unpack *u) {
    OC(post_recv(&u->hpp, &c->pp));
}

void post_send(Pack *p, Comm *c) {
    UC(post_send(&p->hpp, &c->pp));
}

void wait_recv(Comm *c, Unpack *u) {
    UC(wait_recv(&c->pp, &u->hpp));
}

void wait_send(Comm *c) {
    wait_send(&c->pp);
}


/* momentum */

void post_recv(CommM *c, UnpackM *u) {
    OC(post_recv(&u->hmm, &c->mm));
    OC(post_recv(&u->hii, &c->ii));
}

void post_send(PackM *p, CommM *c) {
    UC(post_send(&p->hmm, &c->mm));
    UC(post_send(&p->hii, &c->ii));
}

void wait_recv(CommM *c, UnpackM *u) {
    UC(wait_recv(&c->mm, &u->hmm));
    UC(wait_recv(&c->ii, &u->hii));
}

void wait_send(CommM *c) {
    wait_send(&c->mm);
    wait_send(&c->ii);
}
