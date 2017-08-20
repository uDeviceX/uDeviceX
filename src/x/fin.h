namespace x {
void fin() {
    rex::fin();
    Pfree0(buf_pi);

    fin_ticketcom(tc);
    fin_ticketpack(tp);
    fin_ticketpinned(ti);
}
}
