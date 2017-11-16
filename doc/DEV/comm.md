# comm

generic communicator with "halo".

## purpose

## data structures

* `hBags`: buffers on the host, contains all necessary information of the data to communicate:
  * `data`: host buffer containing the data
  * `counts`: number of items per fragment
  * `capacity`: maximum number of items of buffers `data`
  * `bsize`: size (in bytes) of one item
* `dBags`: buffers on the device
* `Stamp`: contains the communication related variables:
  * `sreq`, `rreq`: send and receive requests
  * `bt`: base tag: tag of one exchange is `bt + fid`, where `fid` is the fragment id
  * `cart`: cartesian communicator
  * `ranks`: ranks of the neighbors in the grid (who do I send to?)
  * `tags`: tags used by neighbors to send messages
  
