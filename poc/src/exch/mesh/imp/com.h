void emesh_post_recv(EMeshComm *c, EMeshUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->pp));
}

void emesh_post_send(EMeshPack *p, EMeshComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->pp));
}

void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u) {
    UC(comm_wait_recv(c->pp, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void emesh_wait_send(EMeshComm *c) {
    UC(comm_wait_send(c->pp));
}


/* momentum */

void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void emesh_post_send(EMeshPackM *p, EMeshCommM *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void emesh_wait_send(EMeshCommM *c) {
    UC(comm_wait_send(c->comm));
}
