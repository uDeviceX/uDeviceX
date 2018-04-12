void sampler_data_ini(SampleData **s) {
    EMALLOC(1, s);
    sampler_data_reset(*s);
}

void sampler_data_fin(SampleData  *s) {
    EFREE(s);
}

void sampler_data_reset(SampleData *s) {
    s->n = 0;
}

void sampler_data_push(long n, const Particle *pp, const float *ss, SampleData *s) {
    int id;
    SampleDatum *d;
    id = s->n++;

    if (s->n >= MAX_N_DATA)
        ERR("Too many data: %d/%d", s->n / MAX_N_DATA);

    d = s->d + id;
    d->n = n;
    d->pp = pp;
    d->ss = ss;
}

