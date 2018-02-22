void time_step_accel_ini(/**/ TimeStepAccel **pq) {
    TimeStepAccel *q;
    EMALLOC(1, &q);
    q->k = 0;
    *pq = q;
}
void time_step_accel_fin(TimeStepAccel *q) { EFREE(q); }
void time_step_accel_push(TimeStepAccel *q, float m, int n, Force *ff) {
    int k;
    k = q->k;
    if (k + 1 == MAX_ACCEL_NUM)
        ERR("k + 1 = %d < %d = MAX_ACCEL_NUM", k + 1, MAX_ACCEL_NUM);
    if (m <= 0) ERR("m <= 0");
    q->mm[k] = m; q->nn[k] = n; q->fff[k] = ff;
    q->k = k + 1;
}
void time_step_accel_reset(TimeStepAccel *q) { q->k = 0; }

void time_step_ini(Config *c, /**/ TimeStep **pq) {
    const char *type;
     tlog[q->type](q); }
float time_step_dt0(TimeStep *q) { return q->dt; };
