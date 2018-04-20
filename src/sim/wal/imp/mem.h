void wall_ini(const Config *cfg, int3 L, Wall **wall) {
    Wall *w;
    EMALLOC(1, wall);
    w = *wall;
    UC(sdf_ini(L, &w->sdf));
    UC(wall_ini_quants(L, &w->q));
    UC(wall_ini_ticket(L, &w->t));
    UC(wvel_ini(&w->vel));
    UC(wvel_set_conf(cfg, w->vel));
    UC(wvel_step_ini(&w->velstep));
}

void wall_fin(Wall *w) {
    UC(sdf_fin(w->sdf));
    UC(wall_fin_quants(&w->q));
    UC(wall_fin_ticket(w->t));
    UC(wvel_fin(w->vel));
    UC(wvel_step_fin(w->velstep));
    EFREE(w);
}
