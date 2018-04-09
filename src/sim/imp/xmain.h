static void gen_flu(Sim *s) {
    Flu *flu = &s->flu;
    UC(flu_gen_quants(s->coords, s->params.numdensity, s->gen_color, &flu->q));
    UC(flu_build_cells(&flu->q));
    if (opt->fluids) flu_gen_ids(s->cart, flu->q.n, &flu->q);
}

static void set_options_flu_only(Sim *s) {
    Opt *o = &s->opt;
    o->rbc = false;
    o->rig = false;
    o->wall = false;
}

void sim_gen_flu(Sim *s) {
    gen_flu(s);
    set_options_flu_only(s);
    dSync();
    MC(m::Barrier(s->cart));
}
