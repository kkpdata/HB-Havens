import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets

from hbhavens.ui import widgets


class SplittedTab(widgets.AbstractTabWidget):
    """
    Widget with the background map

    This widget works based on a WMTS, which contains background tiles
    within and around the axes limits. Each action can update the bounding
    box, and thus the tile set.
    """

    def __init__(self, mainwindow, leftwidget, rightwidget):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, mainwindow)
        
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        self.splitter.addWidget(leftwidget)
        self.splitter.addWidget(rightwidget)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.splitter)

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)

        self.splitter.splitterMoved.connect(self.on_moved)
        
        handle_layout = QtWidgets.QVBoxLayout()
        handle_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.setHandleWidth(12)

        self.button = QtWidgets.QToolButton()
        self.button.setStyleSheet('background-color: rgba(255, 255, 255, 0)')
        

        rightlogo = os.path.join(self.mainmodel.datadir, 'icons', 'iconfinder_icon-chevron-right_211647.png')
        leftlogo = os.path.join(self.mainmodel.datadir, 'icons', 'iconfinder_icon-chevron-left_211647.png')
        self.righticon = QtGui.QIcon(rightlogo)
        self.lefticon = QtGui.QIcon(leftlogo)
        self.icon = self.righticon
        
        self.button.setIcon(self.icon)
        
        self.button.clicked.connect(self.handleSplitterButton)

        handle_layout.addWidget(self.button)

        handle_layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding))

        handle = self.splitter.handle(1)
        handle.setLayout(handle_layout)

        self.setLayout(layout)

    def handleSplitterButton(self, left=True):
    
        # If one is collapsed
        if not all(self.splitter.sizes()):
            # Open and set right icon
            self.splitter.setSizes([1, 1])
        else:
            self.splitter.setSizes([1, 0])
            
        self.on_moved()

    def on_moved(self):
        """Method to call when splitter is moved. This method
        updates the icon, but it can be extended to update for example a canvs
        """

        # If one is collapsed
        if not all(self.splitter.sizes()):
            # Open and set right icon
            icon = self.lefticon
        # If both are open
        else:
            icon = self.righticon
        
        if self.icon is not icon:
            self.icon = icon
            self.button.setIcon(self.icon)
        
    

class InteractiveLegend:

    def __init__(self, widget, elements, loc='upper right', title=''):
        self.handles = {}
        self.labels = {}
        self.lined = {}
        self.widget = widget
        self.elements = elements
        self.loc = loc
        self.title = title

    def add_item(self, element, handle, label):
        self.handles[element] = handle
        self.labels[element] = label
        
    def remove(self, element):
        del self.handles[element]
        del self.labels[element]

    def _onpick(self, event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        handle = event.artist
        collection = self.lined[handle]
        alpha = collection.get_alpha()

        vis = not collection.get_visible()
        collection.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            handle.set_alpha(alpha)
        else:
            handle.set_alpha(alpha / 3.)
    
        self.widget.canvas.draw()
    
    def _update_legend(self):
        
        self.lined.clear()

        keys = list(self.elements.keys())
        if not keys:
            self.widget.ax.legend([], [], fancybox=False, edgecolor='0.2', fontsize=8, labelspacing=0.8, borderpad=0.7, loc=self.loc, title=self.title)
            return None

        # Collect the handles and labels and construct legend.
        self.legend = self.widget.ax.legend(
            [self.handles[key] for key in keys],
            [self.labels[key] for key in keys],
            fancybox=False, edgecolor='0.2', fontsize=8, labelspacing=0.8, borderpad=0.7, loc=self.loc, title=self.title
        )
        self.legend.get_frame().set_linewidth(0.5)

        # Note that the collected handles are not the handles which will be in
        # the legend, so they need to be activated for picking seperately
        for key, handle in zip(keys, self.legend.legendHandles):
            # Enable picking, 5 pts tolerance
            handle.set_picker(5)
            self.lined[handle] = self.elements[key]


