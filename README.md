# Content:

* Fuzzy C-Means (FCM); Fuzzy J-Means (FJM) and FVNS (VARIABLE NEIGHBORHOOD SEARCH METAHEURISTICS FOR FCM and FJM)
* Various cluster validity indices:
    * Partition Coefficient (PC)
    * Normalized Partition Coefficient (NPC)
    * Fuzzy Hypervolume (FHV)
    * Fukuyama-Sugeno Index (FS)
    * Xie-Beni Index (XB)
    * Beringer-Hullermeier Index (BH)
    * Bouguessa-Wang-Sun index (BWS)
* For cluster validity indices and for ploting fuzzy clustering we have used packages from:
       https://github.com/99991/FuzzyClustering

# Fuzzy J-Means Algorithm
Fuzzy J-Means (FJM) is python module implementing the clustering methods based on fuzzy j-means heuristic (FJM) and fvns the integration of FJM to variable neighborhood search metaheuristic.
FJM is a local search heuristic is inspired from (Hansen and Mladenovic, 2001), where moves belong to the neighbourhood of the current solution defined by all possible centroid-to-pattern relocations (Belacel et al. 2002). This crisp solution found is then transformed into a fuzzy one by an alternate step, i.e., by finding centroids and membership degrees for all patterns and clusters. Like the F-CM method, the F-JM heuristic may be stuck in local minima, possibly of poor value. To alleviate this difficulty, the F-JM heuristic is embedded into the variable neighbourhood search (VNS) metaheuristic.
## Prerequisites
* Python 3.7+
## Usage
    python fvjm.py ./datasets/"file".csv
    * For example: 
    python fvjm.py ./datasets/iris.csv

The following parameters can be set using the file fvjm_init.py: 

* NUM_OF_CLUSTER = 6 
* FUZZY_FACTOR = 2  (m >0, default m = 2)
* TERMINATION_THRESHOLD = 0.0001
* MAX_CPU_TIME = 10 "the time as stop criteria for metaheuristic VNS"
* OUT_DIR = ''
* SMALL = 0.0001
* OUTPUT_DIR = 'results'

## Citation:
If you use our Fjmeans in a scientific publication, we would appreciate using the following citations:

```
@article{belacel2002fuzzy,
  title={Fuzzy J-means: a new heuristic for fuzzy clustering},
  author={Belacel, Nabil and Hansen, Pierre and Mladenovic, Nenad},
  journal={Pattern recognition},
  volume={35},
  number={10},
  pages={2193--2200},
  year={2002},
  publisher={Elsevier}
}

```
# Publications

Shakirova, A., Nichman, L., Belacel, N., Nguyen, C., Bliankinshtein, N., Wolde, M., ... & Huang, Y. (2022). Multivariable Characterization of Atmospheric Environment with Data Collected in Flight. Atmosphere, 13(10), 1715.

Belacel, Nabil, Miroslava Čuperlović-Culf, Mark Laflamme, and Rodney Ouellette. "Fuzzy J-Means and VNS methods for clustering genes from microarray data." Bioinformatics 20, no. 11 (2004): 1690-1701.

Belacel, N., Hansen P., Mladenovic N. (2002) "Fuzzy J-means: a new heuristic for fuzzy clustering." Pattern recognition 35.10 (2002): 2193-2200.

Hansen P., Mladenovic N. (2001) J-Means: a new local search heuristic for minimum sum of squares clustering, Pattern Recognition, Volume 34, Issue 2, Pages 405-413,

# Acknowledgements

Thanks to Zhicheng (Max) Xu for his contribution in converting c++ code of FJM to python.

The FJM algorithm initially was developed during N. Belacel's post-doc in GERAD University of Montreal and under supervision of wounderful Professors Pierre Hansen and Nenad Mladenovic.
