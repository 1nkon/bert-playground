To install dependencies:  
`$ bin/install_dependencies.sh`  
To start server:  
`$ bin/start_server.sh`  

Server has single route in root which accepts POST json requests with parameters: 
* "sentence": Target sentence. Word to find synonyms for must be marked with `#`
* "simple" (Optional): If set to true omits verbose data.  

Simple request could be made with a script, e.g.  
`$ bin/predict.sh "what is the chance of rain# tomorrow?"`