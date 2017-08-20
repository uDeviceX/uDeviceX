namespace x {
void fin() {
    rex::fin();
    cudaFree(buf);
    Pfree(buf_pi);
    fin_ticketcom(tc);
    fin_ticketpack(tp);
    fin_ticketpinned(ti);
}
}
