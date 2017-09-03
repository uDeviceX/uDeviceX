namespace x {
static void ini_tickets(/*io*/ basetags::TagGen *g) {
    first = 1;
    ini_ticketcom(&tc);
    ini_ticketr(&tr);
    ini_tickettags(g, &tt);
    ini_ticketpack(&tp);
    ini_ticketpinned(&ti);

    Palloc0(&buf_pi, MAX_PART_NUM);
    Link(&buf, buf_pi);
}

static void ini_remote() {
    
}

void ini(/*io*/ basetags::TagGen *g) {
    ini_tickets(g);
    ini_remote();
    rex::ini();
}
}
