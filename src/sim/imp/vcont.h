/* velocity controller */

static bool valid_step(long id, const int freq) {
    return (freq != 0) && (id % freq == 0);
}

void sample(long id, const Flu *f, /**/ PidVCont *c) {
    if (valid_step(id, VCON_SAMPLE_FREQ)) {
        const flu::Quants *q = &f->q;
        sample(q->n, q->pp, q->cells.starts, q->cells.counts, /**/ c);
    }
}

void adjust(long id, /**/ PidVCont *c, scheme::force::Param *fpar) {
    if (valid_step(id, VCON_ADJUST_FREQ)) {
        float3 f;
        f = adjustF(/**/ c);

        fpar->a = f.x;
        fpar->b = f.y;
        fpar->c = f.z;
    }
}

void log(long id, const PidVCont *c) {
    if (valid_step(id, VCON_LOG_FREQ))
        log(c);
}
