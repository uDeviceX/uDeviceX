namespace x {
void ini(/*io*/ basetags::TagGen *tg) { rex::ini(tg); }
void fin() { rex::fin(); }

void rex(std::vector<ParticlesWrap> w) {
    rex::bind_solutes(w);
    rex::pack_p();
    rex::post_p();
    rex::recv_p();

    rex::halo(); /* fsi::halo(); */

    rex::post_f();
    rex::recv_f();
}
}
