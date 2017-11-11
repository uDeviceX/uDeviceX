void fin_ticket(TicketT *t) {
    destroy(&t->textri);
    destroy(&t->texadj0);
    destroy(&t->texadj1);
    destroy(&t->texvert);
}
