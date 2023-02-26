import GenerateHalo as gh
import GenerateCluster as gc
import io as io


#Read in data
data = io.Read()

#Generate haloes
haloes = gh()

#Generate clusters
clusters = gc(haloes)

#Generate Summary Visualizations
visualizations = gv()