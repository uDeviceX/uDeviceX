/* ini */
void dflu_status_ini(DFluStatus **s);
void dflu_status_fin(DFluStatus  *s);

/* get */
int  dflu_status_success(DFluStatus *s);
void dflu_status_log(DFluStatus *s);

/* set */
void dflu_status_over(int count, int cap, /**/ DFluStatus *s);
