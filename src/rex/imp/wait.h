namespace sub {
void wait(std::vector<MPI_Request> &v) {
    MPI_Status s[v.size()];
    if (v.size()) MC(m::Waitall(v.size(), &v.front(), s));
    v.clear();
}

namespace s { /* send */
void waitC() { wait(reqsendC); }
void waitP() { wait(reqsendP); }
void waitA() { wait(reqsendA); }
}

namespace r { /* recive */
void waitC() { wait(reqrecvC); }
void waitP() { wait(reqrecvP); }
void waitA() { wait(reqrecvA); }
}
}
