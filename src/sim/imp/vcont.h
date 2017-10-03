/* velocity controller */

static bool valid_step(long id, const int freq) {
    return (freq != 0) && (id % freq == 0);
}

void sample(long id, int n, const Particle *pp, const int *starts, /**/ PidVCont *c) {
    if (valid_step(id, VCON_SAMPLE_FREQ))
        sample(n, pp, starts, /**/ c);
}

void adjust(long id, /**/ PidVCont *c, scheme::Fparams *fpar) {
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
