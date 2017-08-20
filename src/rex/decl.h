namespace rex {
std::vector<MPI_Request> reqsendC, reqrecvC, reqsendP, reqrecvP, reqsendA, reqrecvA;
RemoteHalo remote[26];
LocalHalo  local[26];
}
