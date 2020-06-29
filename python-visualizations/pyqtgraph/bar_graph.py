import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from sklearn.datasets import load_wine


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    pg_win = pg.GraphicsLayoutWidget()
    
    dataset = load_wine()
    data = dataset['data']
    features = dataset["feature_names"]
    
    attributes = features[:5]
    
    df = pd.DataFrame(data, columns=features)
    
    plots = []
    
    for a, attribute in enumerate(attributes):
        y, x = np.histogram(df[attribute])
        plot = pg_win.addPlot(name=attribute, title=attribute)
        plot.plot(x, y, stepMode=True, fillLevel=0, fillOutline=True, brush=(0, 0, 255, 128))
        plot.setLabel("bottom", "Values")
        if a == 0:
            plot.setLabel("left", "Counts")
        plots.append(plot)
    
    for plot in plots[1:]:
        plot.setYLink(plots[0])
    
    pg_win.show()
    sys.exit(app.exec_())