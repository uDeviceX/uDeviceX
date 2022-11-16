void wall_repulse_prm_ini(float lambda, WallRepulsePrm **wrp) {
    WallRepulsePrm *wr;
    EMALLOC(1, wrp);
    wr = *wrp;
    wr->l = lambda;
}

void wall_repulse_prm_ini_conf(const Config *cfg, const char *ns, WallRepulsePrm **wrp) {
    float lambda;
    UC(conf_lookup_float_ns(cfg, ns, "lambda", &lambda));
    UC(wall_repulse_prm_ini(lambda, wrp));
}

void wall_repulse_prm_fin(WallRepulsePrm *wr) {
    EFREE(wr);
}
