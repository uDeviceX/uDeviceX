namespace cnt {
void fin() {
    delete indexes;
    scan::free_work(&ws);
    delete entries;
    delete starts;
    delete  counts;
    delete rgen;
}
}
