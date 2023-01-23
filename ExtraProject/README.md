### Additional Exercise: Bias-Variance Tradeoff
In the additional exercise we did a Bias-Variance Tradeoff analysis on terrain data from Norway.
We looked at the trade of for three linear regression models (OLS, Ridge, Lasso), a feed forward neural network, and a decision tree.

The code used is found in `terrain.py` and `functions.py`, whereas the first contains the
code with computation and plotting, and functions contains some functions used.
The data was obtained from https://earthexplorer.usgs.gov/, and it contains altitude of terrain. 
`SRTM_data_Norway_1.tif` is the file containing the data.
We looked at a small 50x50 area to reduce computation time.

**How to use**:
Two uncommon modules were used in this code. The *mlxtend* module for calculating variance, bias and loss,
and *tqmd* for displaying progressbar in terminal during bootstrap. Other dependencies are numpy, matplotlib, sklearn, seaborn, imageio, inpsect, warnings and importlib. 

Running the file 
```python
python3 terrain.py
```
will plot the surface the models are fitting.
Unncomment the different functions "BVT_..." to see various bias-variance tradeoffs.
##
