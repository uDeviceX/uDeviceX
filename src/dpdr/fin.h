void fin_tcom(/**/ MPI_Comm *cart) {
    MC(l::m::Comm_free(cart));
}

void fin_trnd() {}
