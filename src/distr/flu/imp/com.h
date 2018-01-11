void dflu_post_recv(DFluComm *c, DFluUnpack *u) {
    UC(post_recv(&u->hpp, c->pp));
    if (global_ids)    UC(post_recv(&u->hii, c->ii));
    if (multi_solvent) UC(post_recv(&u->hcc, c->cc));
}

void dflu_post_send(DFluPack *p, DFluComm *c) {
    UC(post_send(&p->hpp, c->pp));
    if (global_ids)    UC(post_send(&p->hii, c->ii));
    if (multi_solvent) UC(post_send(&p->hcc, c->cc));
}

void dflu_wait_recv(DFluComm *c, DFluUnpack *u) {
    UC(wait_recv(c->pp, &u->hpp));
    if (global_ids)    UC(wait_recv(c->ii, &u->hii));
    if (multi_solvent) UC(wait_recv(c->cc, &u->hcc));
}

void dflu_wait_send(DFluComm *c) {
    UC(wait_send(c->pp));
    if (global_ids)    UC(wait_send(c->ii));
    if (multi_solvent) UC(wait_send(c->cc));
}
