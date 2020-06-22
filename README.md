# ConvCN

This Github repo stores code used in the paper: ConvCN: Enhancing Graph Based Citation Recommendation Using Citation Network Embedding

Step to reproduce experiment result

For Citation Network Embedding Experiment

   1. run train_ConvCN_Citation_Network_Embedding.py
   2. run eval_Citation_Network_embedding.py

For Citation Recommendation Experiment

   1. run train_ConvCN_Citation_Recommendation.py
   2. run code in PaperRank.ipynb to get recommendation result from PaperRank
   3. run code in CF method directory (just run Collaborative Filtering.ipynb by using user-based method) to get recommendation  result from CF
   3. run eval_ConvCN_CF_weighted_avg.py (for ConvCN-CF model) and eval_ConvCN_PaperRank.py (for ConvCN-PR model)

Note that data can be found in Dataset directory.

I recommend to run these code in GPU.
