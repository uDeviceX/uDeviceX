namespace rex {
int recv_counts[26], send_counts[26];

std::vector<MPI_Request> reqsendC, reqrecvC, reqsendP, reqrecvP, reqsendA, reqrecvA;
RemoteHalo remote[26];
LocalHalo  local[26];
}
