void drbc_post_recv(DRbcComm *c, DRbcUnpack *u) {
    UC(post_recv(&u->hpp, c->pp));
    if (rbc_ids) UC(post_recv(&u->hii, c->ii));
}

void drbc_post_send(DRbcPack *p, DRbcComm *c) {
    UC(post_send(&p->hpp, c->pp));
    if (rbc_ids) UC(post_send(&p->hii, c->ii));
}

void drbc_wait_recv(DRbcComm *c, DRbcUnpack *u) {
    UC(wait_recv(c->pp, &u->hpp));
    if (rbc_ids) UC(wait_recv(c->ii, &u->hii));
}

void drbc_wait_send(DRbcComm *c) {
    UC(wait_send(c->pp));
    if (rbc_ids) UC(wait_send(c->ii));
}
