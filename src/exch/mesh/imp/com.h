void post_recv(Comm *c, Unpack *u) {
    post_recv(&u->hpp, &c->pp);
}

void post_send(Pack *p, Comm *c) {
    post_send(&p->hpp, &c->pp);
}

void wait_recv(Comm *c, Unpack *u) {
    wait_recv(&c->pp, &u->hpp);
}

void wait_send(Comm *c) {
    wait_send(&c->pp);
}


/* momentum */

void post_recv(CommM *c, UnpackM *u) {
    post_recv(&u->hmm, &c->mm);
    post_recv(&u->hii, &c->ii);
}

void post_send(PackM *p, CommM *c) {
    post_send(&p->hmm, &c->mm);
    post_send(&p->hii, &c->ii);
}

void wait_recv(CommM *c, UnpackM *u) {
    wait_recv(&c->mm, &u->hmm);
    wait_recv(&c->ii, &u->hii);
}

void wait_send(CommM *c) {
    wait_send(&c->mm);
    wait_send(&c->ii);
}
