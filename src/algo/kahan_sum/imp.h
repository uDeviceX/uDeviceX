struct KahanSum;

void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input);
void kahan_sum_get(KahanSum*);
