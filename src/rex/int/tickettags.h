namespace rex {
/* ticket recive */
static void ini_tickettags(/*io*/ basetags::TagGen *tg, /*o*/ TicketTags *t) {
    t->btc  = get_tag(tg);
    t->btp1 = get_tag(tg);
    t->btp2 = get_tag(tg);
    t->btf  = get_tag(tg);
}
}
