namespace x {
void ini(/*io*/ basetags::TagGen *tg) { rex::ini(tg); }
void fin() { rex::fin(); }

void rex0(std::vector<ParticlesWrap> w, int nw) {
    rex::pack_p(nw);
    rex::_pack_attempt(w);
    rex::post_p(w);
    rex::recv_p(w);
    rex::halo(); /* fsi::halo(); */
    rex::_postrecvP();
    rex::post_f(w);
    rex::recv_f(w);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
