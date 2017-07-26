namespace basetags {
struct TagGen {
    int bt = 0;
    static const int stride = 100;
};

inline void ini(TagGen *tg)    {tg->bt = 0;}
inline int get_tag(TagGen *tg) {return tg->bt += tg->stride;}
}
