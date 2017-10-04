void free_quants(Quants *q) {
    CC(cudaFree(q->pp)); CC(cudaFree(q->pp0));
    fin(&q->cells);
    fin_ticket(&q->tcells);
    delete[] q->pp_hst;
}

void free_quantsI(QuantsI *q) {
    CC(cudaFree(q->ii)); CC(cudaFree(q->ii0));
    delete[] q->ii_hst;
}

void free_ticketZ(/**/ TicketZ *t) {
    float4  *zip0 = t->zip0;
    ushort4 *zip1 = t->zip1;
    cudaFree(zip0);
    cudaFree(zip1);
}

void free_ticketRND(/**/ TicketRND *t) {
    delete t->rnd;
}
