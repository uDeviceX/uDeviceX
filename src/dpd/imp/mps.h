static void mps() {
    is_mps_enabled = false;

    const char * mps_variables[] = {
        "CRAY_CUDA_MPS",
        "CUDA_MPS",
        "CRAY_CUDA_PROXY",
        "CUDA_PROXY"
    };
    
    for(int i = 0; i < 4; ++i)
        is_mps_enabled |= getenv(mps_variables[i])!= NULL && atoi(getenv(mps_variables[i])) != 0;
}
