To install dependencies:  
`$ bin/install_dependencies.sh`  
To start server:  
`$ bin/start_server.sh`  

Server has single route in root which accepts POST json requests with parameters: 
* "sentence": Target sentence. Word to find synonyms for must be marked with `#`
* "lower", "upper" (Optional, substitute marking with `#`): Positions in the sentence of start and end of collocation to find synonyms for.  
Note: sentence is split via whitespaces, upper bound is inclusive. 
* "simple" (Optional): If set to true omits verbose data.  

Simple request could be made with a script, e.g.  
`$ bin/predict.sh "what is the chance of rain# tomorrow?"`