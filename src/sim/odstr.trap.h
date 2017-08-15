void odstr() {
    int i;
    for(i=0; ; ++i){
        odistr_trap::odstr();
        if (i%10000==0) MSG("i=%d", i);
    }
}
