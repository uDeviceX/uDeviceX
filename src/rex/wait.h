namespace rex {
void wait(std::vector<MPI_Request> &v) {
    MPI_Status statuses[v.size()];
    if (v.size()) MC(l::m::Waitall(v.size(), &v.front(), statuses));
    v.clear();
}

void waitC() { wait(reqsendC); }
void waitP() { wait(reqsendP); }
void waitA() { wait(reqsendA); }
}
