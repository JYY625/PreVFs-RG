# PreVFs-RG
**PreVFs-RG: A deep hybrid model for identifying virulence factors based on residual block and gated recurrent unit.**
# Usage
1.The data file contains training dataset and independent dataset.  
2.kmer, DDE and AAC are implementation of feature extraction.  
3.The integrating of residual block and gated recurrent unit (GRU) are as the classifier.  

 Configuration Environment：python=3.9, tensorflow=2.7.0, keras=2.7.0, numpy=1.21.0.  
 
 Firstly, we used Kmer, DDE and AAC to extract features.  
 And the deep model is constructed by integrating residual block and gated recurrent unit (GRU).  
 Finally, the prediction is verified by 10-fold cross-validation.
 
