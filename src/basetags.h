namespace basetags {
struct TagGen {
    int bt = 0;
    static const int stride = 100;
};

void ini(TagGen *tg)    {tg->bt = 0;}
int get_tag(TagGen *tg) {return tg->bt += tg->stride;}
}
