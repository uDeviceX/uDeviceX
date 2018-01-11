void emesh_post_recv(EMeshComm *c, EMeshUnpack *u) {
    UC(post_recv(&u->hpp, &c->pp));
}

void emesh_post_send(EMeshPack *p, EMeshComm *c) {
    UC(post_send(&p->hpp, &c->pp));
}

void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u) {
    UC(wait_recv(&c->pp, &u->hpp));
}

void emesh_wait_send(EMeshComm *c) {
    UC(wait_send(&c->pp));
}


/* momentum */

void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u) {
    UC(post_recv(&u->hmm, &c->mm));
    UC(post_recv(&u->hii, &c->ii));
}

void emesh_post_send(EMeshPackM *p, EMeshCommM *c) {
    UC(post_send(&p->hmm, &c->mm));
    UC(post_send(&p->hii, &c->ii));
}

void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u) {
    UC(wait_recv(&c->mm, &u->hmm));
    UC(wait_recv(&c->ii, &u->hii));
}

void emesh_wait_send(EMeshCommM *c) {
    UC(wait_send(&c->mm));
    UC(wait_send(&c->ii));
}
