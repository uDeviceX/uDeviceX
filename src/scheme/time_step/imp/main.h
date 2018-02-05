void time_step_accel_ini(/**/ TimeStepAccel **pq) {
    TimeStepAccel *q;
    EMALLOC(1, &q);
    *pq = q;
}
void time_step_accel_fin(TimeStepAccel *q) { EFREE(q); }
