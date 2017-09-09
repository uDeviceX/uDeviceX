namespace odstr {
namespace sub {

void fin_S(Send *s) {
    for(int i = 0; i < 27; ++i) Dfree(s->iidx_[i]);
    Dfree(s->iidx);
    
    Dfree(s->pp.dp[0]); /* r.pp.dp[0] = s.pp.dp[0] */
    dealloc(&s->pp);
    
    Dfree(s->size_dev); Dfree(s->strt);
    dual::dealloc(s->size_pin);
}

void fin_R(Recv *r) {
    dealloc(&r->pp);
    Dfree(r->strt);
}

void fin_SRI(Pbufs<int> *sii, Pbufs<int> *rii) {
    Dfree(sii->dp[0]); /* rii.dp[0] = sii.dp[0] */
    dealloc(sii);
    dealloc(rii);
}

} // sub
} // odstr
