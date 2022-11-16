void field_fin(Field *q) {
    EFREE(q->D);
    EFREE(q);
}
