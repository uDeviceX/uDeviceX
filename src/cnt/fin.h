namespace cnt {
void fin() {
    delete subindices;
    scan::free_work(&ws);
    delete cellsentries;
    delete cellsstart;
    delete  cellscount;
    delete rgen;
}
}
