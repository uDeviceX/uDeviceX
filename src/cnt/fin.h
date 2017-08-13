namespace cnt {
void fin() {
    delete indexes;
    scan::free_work(&ws);
    delete entries;
    delete  counts;
    delete rgen;
    
    Pfree(starts);
}
}
