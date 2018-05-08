## Directory structure
- **css**: Webpage style  
- **tools**: Utility tools  
- **make**: Building files  
- **src**: Documentation files  
- **include**: Included files (e.g. pictures)  
- **plots**: Source files to generate included files  
- *index.adoc*: Main documentation file


## Necessary packages
In order to modify the documentation,
the following packages should already be installed:  
- asciidoctor  
- pygments.rb  
- General uDeviceX tools: `(cd ..; make install)`  
- documentation tools: `make -C tools`


## Modifying existing files
1. `make`  
2. `./tools/start` -> starts server. do it only once  
3. `./tools/view` -> to view changes locally  
4. add/commit/push to udx repo  
5. `./tools/deploy` -> updates online documentation (ssh key in github account must me set!)  


## Adding new files
- run `./configure`  
- follow the procedure for modifying files
