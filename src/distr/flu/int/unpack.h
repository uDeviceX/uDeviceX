void unpack_pp(/**/ Unpack *u) {
    int nhalo;
    nhalo = unpack_pp(u->hpp, /**/ u->ppre);
    u->nhalo = nhalo;
}

void unpack_ii(/**/ Unpack *u) {
    unpack_ii(u->hii, /**/ u->iire);
}

void unpack_cc(/**/ Unpack *u) {
    unpack_ii(u->hcc, /**/ u->ccre);
}
