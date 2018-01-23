void dflu_post_recv(DFluComm *c, DFluUnpack *u) {
    UC(comm_post_recv(&u->hpp, c->pp));
    if (global_ids)    UC(comm_post_recv(&u->hii, c->ii));
    if (multi_solvent) UC(comm_post_recv(&u->hcc, c->cc));
}

void dflu_post_send(DFluPack *p, DFluComm *c) {
    UC(comm_post_send(&p->hpp, c->pp));
    if (global_ids)    UC(comm_post_send(&p->hii, c->ii));
    if (multi_solvent) UC(comm_post_send(&p->hcc, c->cc));
}

void dflu_wait_recv(DFluComm *c, DFluUnpack *u) {
    UC(comm_wait_recv(c->pp, &u->hpp));
    if (global_ids)    UC(comm_wait_recv(c->ii, &u->hii));
    if (multi_solvent) UC(comm_wait_recv(c->cc, &u->hcc));
}

void dflu_wait_send(DFluComm *c) {
    UC(comm_wait_send(c->pp));
    if (global_ids)    UC(comm_wait_send(c->ii));
    if (multi_solvent) UC(comm_wait_send(c->cc));
}
