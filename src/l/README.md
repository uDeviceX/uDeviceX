# [l]ibraries

libraries which do not depend on [conf.h](../conf.h).

* [inc](../inc) headers
* [h](h) [h]ost
* [d](d) [d]evice

Headers are included in [bund.cu](../bund.cu) as 

    namespace l {
      #include "inc/<libname>.h"
    }
