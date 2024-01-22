# -*- coding: utf-8 -*-
"""
Created on  : Thu Aug 24 16:49:34 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : PR3594.10.00
Description : HB Havens ui custom models
"""

import logging
from collections import namedtuple
from io import BytesIO
from multiprocessing.pool import ThreadPool
import bs4
import matplotlib.pyplot as plt
import numpy as np
import owslib.tms
import owslib.wmts
import requests
import xmltodict
from PyQt5 import Qt, QtCore, QtGui

logger = logging.getLogger(__name__)

class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

        self.sort_column = None
        self.sort_order = False
        self.flip_count = 0

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iat[index.row(), index.column()]).replace('nan', '')
            if role == QtCore.Qt.TextAlignmentRole:
                return QtCore.QVariant(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        return None

    def headerData(self, rowcol, orientation, role):
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return str(self._data.index[rowcol])
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return str(self._data.columns[rowcol])
        return None


    def _administrate_order(self):
        """
        Administrate the sorting order. Registers the clicked_column, and checks
        if the order should be switched. Since each click action consists of
        two events, only one is administrated. The other will be skipped.

        Returns
        -------
        execute : bool
            Boolean which indicates if the sorting should be executed.
        """

        # Each user mouse clock is registeed twice, but we only want to sort
        # once
        self.flip_count = (self.flip_count + 1) % 2
        if self.flip_count == 1:
            return False

        self.sort_order = not self.sort_order

        return True

    def get_clicked_column_name(self, clicked_column):
        """
        Get clicked column name.

        Parameters
        ---------
        clicked_column : int
            number of the column that is clicked

        """
        return self._data.columns[clicked_column]

    def sort(self, clicked_column, order):
        """Sort table by given column number."""

        # Check if sorting should be executed
        execute = self._administrate_order()
        if not execute:
            return None

        self.layoutAboutToBeChanged.emit()

        # Actually sort the data
        # indexname = self._data.index.name
        self.sort_column = self.get_clicked_column_name(clicked_column)
        # Reset and set the index so the index can be used in sorting.
        # self._data = self._data.reset_index().sort_values(sort_columns, ascending=[self.sort_order, True]).set_index(indexname)
        # self._data.sort_index(inplace=True, ascending=True)
        self._data.sort_values(by=self.sort_column, ascending=self.sort_order, inplace=True)

        self.layoutChanged.emit()

class PandasModelSelection(PandasModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None, column_selection=None):
        # Check columns
        for col in column_selection:
            if col not in data.columns:
                logger.warning('Column "{}" is not present in dataframe'.format(col))
                column_selection.remove(col)

        PandasModel.__init__(self, data)
        self.column_selection = []
        self.columns_conv = {}
        self.parent = parent
        self.set_column_selection(column_selection)
        
    def set_column_selection(self, column_selection):
        # Empty
        del self.column_selection[:]
        self.columns_conv.clear()
        # Assing
        self.column_selection.extend(column_selection)
        self.columns_conv = {column_selection.index(col): imain for imain, col in enumerate(self._data.columns) if col in column_selection}
        self.parent.set_header_sizes()

    def columnCount(self, parent=None):
        return len(self.column_selection)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iat[index.row(), self.columns_conv[index.column()]]).replace('nan', '')
            if role == QtCore.Qt.TextAlignmentRole:
                return QtCore.QVariant(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        return None

    def headerData(self, rowcol, orientation, role):
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return str(self._data.index[rowcol])
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return str(self.column_selection[rowcol])
        return None

    def get_clicked_column_name(self, clicked_column):
        """
        Get clicked column name.

        Parameters
        ---------
        clicked_column : int
            number of the column that is clicked
        """
        return self.column_selection[clicked_column]

class PandasModelEditable(PandasModel):
    """
    Class to populate a table view with a pandas dataframe. This child also
    has editing enabled
    """
    def __init__(self, data, parent=None):
        PandasModel.__init__(self, data, parent)
        self.constraints = {}

    def flags(self, index):
        return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemIsEditable

    def add_constraints(self, col, func, error):
        colnumber = self._data.columns.tolist().index(col)
        self.constraints[colnumber] = (func, error)

    def setData(self, index, value, role=Qt.Qt.EditRole):
        if value == '':
            return True

        row = index.row()
        col = index.column()

        # Check constraints
        if col in self.constraints.keys():
            if not self.constraints[col][0](value):
                raise ValueError('{}: "{}"'.format(self.constraints[col][1], value))

        # Set value
        self._data.iat[row, col] = value
        return True

class WMTS:

    def __init__(self, url, crs, layer):
        """
        Class to connect to WMTS and plot tiles from it.

        Parameters
        ----------
        url : str
            url to wmts
        crs : str
            coordinate reference system
        layer : str
            layer name
        """
        self.url = url
        self.crs = crs
        self.wmts = owslib.wmts.WebMapTileService(url)
        for i, op in enumerate(self.wmts.operations):
            if(not hasattr(op, 'name')):
                self.wmts.operations[i].name = ""

        self.layer = layer
        # Check if layer exists
        layers = self.wmts.contents.keys()
        if layer not in layers:
            raise ValueError('"{}" not in WMTS. Choose from: {}'.format(layer, ', '.join(layers)))

        # Determine interpolation scheme based on layer
        if 'kaart' in self.layer:
            self.interpolation = 'catrom'
        else:
            self.interpolation = 'none'

        self._load_properties_by_scale()

        self.level = 1
        self.tiles = {}
        self.images = {}

        self.is_running = True

    def _load_properties_by_scale(self):
        """
        Method that loads the matrix tile properties per zoom level
        """

        self.matrixproperties = {}
        soup = bs4.BeautifulSoup(self.wmts.getServiceXML().decode(), 'lxml')
        
        for tms in soup.find_all('tilematrixset'):
            # Select crs for tilematrix
            if tms.find('ows:identifier') is None:
                continue

            crs = tms.find('ows:identifier').text
            if crs not in self.crs:
                continue

            for tile in tms.find_all('tilematrix'):
                level = int(tile.find('ows:identifier').text)
                self.matrixproperties[level] = self._get_properties(tile)

        # Determine minimum ans maximum level
        levels = list(self.matrixproperties.keys())
        self.minlevel = min(levels)
        self.maxlevel = max(levels)

    def _get_properties(self, tile):
        """
        Gather the matrixproperties for a given tile, and add to a
        'matrixproperties' namedtuple

        Parameters
        ----------
        tile : xml-object
            tile-object
        """
        MatrixProps = namedtuple('matrixproperties', ['top', 'left', 'xstep', 'ystep', 'matrixwidth',
                                                      'matrixheight', 'tilewidth', 'tileheight', 'pixelsize'])

        left, top = (float(i) for i in tile.topleftcorner.text.strip().split())
        matrixwidth = int(tile.matrixwidth.text)
        matrixheight = int(tile.matrixheight.text)
        tilewidth = int(tile.tilewidth.text)
        tileheight = int(tile.tileheight.text)
        pixelsize = float(tile.scaledenominator.text) * 0.00028

        xstep = tilewidth * pixelsize
        ystep = tileheight * pixelsize

        return MatrixProps(top, left, xstep, ystep, matrixwidth, matrixheight, tilewidth, tileheight, pixelsize)

    def get_tile_indices(self):
        """
        Get the indices of the tiles by extent. The indices depend on the
        bounding box and the level
        """
        indices = namedtuple('indices', ['xmin', 'xmax', 'ymin', 'ymax'])

        # Get matrix properties for the level
        mp = self.matrixproperties[self.level]

        # Determine the x indices
        xmin = int(max(0, (self.bbox[0] - mp.left) // mp.xstep))
        xmax = int(min(mp.matrixwidth, (self.bbox[2] - mp.left) // mp.xstep))

        # Determine the y indices
        ymin = int(max(0, (mp.top - self.bbox[3]) // mp.ystep))
        ymax = int(min(mp.matrixheight, (mp.top - self.bbox[1]) // mp.ystep))

        return indices(xmin, xmax, ymin, ymax)

    def _calculate_level(self, mpp):
        """
        Calculate level based on a certain meter per pixel in the view.
        Correct for minlevel and maxlevel
        """
        level = (11.5 - (np.log2(mpp))).round().astype(int)
        level = max(self.minlevel, min(level, self.maxlevel))
        return level

    def _get_tile(self, args):
        try:
            f = self.wmts.gettile(**args)
            return args['column'], args['row'], plt.imread(BytesIO(f.read()), 0)
        except Exception as e:
            # In can occur that due to a bad connection tiles are not
            # loaded.  In that case, just pass the error and skip the tile.
            logger.error('Failed to load background tile: {}'.format(e))
            return args['column'], args['row'], None      

    def update_tiles(self, bbox, mpp):
        """
        Function to update tiles based on bounding box and level
        """

        newlevel = self._calculate_level(mpp)
        # if newlevel != self.level:
        #     # Clear the figure
        #     for i in list(self.tiles.keys()):
        #         row = self.tiles[i]
        #         for j in list(row.keys()):
        #             self.images[i][j].remove()
        #             del self.images[i][j]
        #             del self.tiles[i][j]

        self.level = newlevel
        self.bbox = bbox

        # Construct tilematrixset name
        tmsname = '{}{}'.format(self.crs, ':' + str(self.level) if self.level is not None else '')

        # Get indices by bbox
        self.indices = self.get_tile_indices()

        arguments = []

        # Get tiles based on indices
        for i in range(self.indices.xmin, self.indices.xmax + 1):
            # Add row if not present yet
            if i not in self.tiles.keys():
                self.tiles[i] = {}

            for j in range(self.indices.ymin, self.indices.ymax + 1):
                # Check if tile is already collected
                if j in self.tiles[i].keys():
                    continue

                # Collect arguments
                arguments.append({
                    'layer': self.layer,
                    'column': i,
                    'row': j,
                    'tilematrixset': self.crs,
                    'tilematrix': tmsname,
                    'format': 'image/png'
                })
                
        # Get the tiles
        tiles = ThreadPool(5).imap_unordered(self._get_tile, arguments)
        for i, j, tile in tiles:
            if tile is not None:
                self.tiles[i][j] = tile

    def plot_and_clean(self, ax, clip):
        """
        This function plots tiles that have not been plotted yet and removes
        parts that are to far from the bounding box.
        """

        mp = self.matrixproperties[self.level]

        for i in list(self.tiles.keys()):
            row = self.tiles[i]
            # If the row is to far from the bounding box, remove.
            if i < (self.indices.xmin - 1) or i > (self.indices.xmax + 1):
                for j, im in self.images[i].items():
                    im.remove()
                del self.images[i]
                del self.tiles[i]
                continue

            # Add row if not present yet
            if i not in self.images.keys():
                self.images[i] = {}

            for j in list(row.keys()):
                tile = row[j]
                # If the tile is to far from the bounding box, remove.
                if j < (self.indices.ymin - 1) or j > (self.indices.ymax + 1):
                    self.images[i][j].remove()
                    del self.images[i][j]
                    del self.tiles[i][j]
                    continue

                # Check if tile is already plotted
                if j in self.images[i].keys():
                    continue

                # plot the image and add to the dictionary
                extent = (
                    mp.left + i * mp.xstep,
                    mp.left + (i + 1) * mp.xstep,
                    mp.top - (j + 1) * mp.ystep,
                    mp.top - j * mp.ystep
                )

                im = ax.imshow(tile, extent=extent, interpolation=self.interpolation)
                self.images[i][j] = im

    def clean_all(self):

        for i in list(self.tiles.keys()):
            for j, im in self.images[i].items():
                im.remove()
            del self.images[i]
            del self.tiles[i]

class TMS:

    def __init__(self, url, crs, layer):
        """
        Class to connect to TMS and plot tiles from it.

        Parameters
        ----------
        url : str
            url to tms
        crs : str
            coordinate reference system
        layer : str
            layer name
        """
        self.url = url
        self.crs = crs
        self.tms = owslib.tms.TileMapService(url+'/tms/1.0.0')
        self.layer = layer
        # Check if layer exists
        # layers = [k.split('/')[-2] for k in self.tms.contents.keys()]
        # if layer not in layers:
            # raise ValueError('"{}" not in WMTS. Choose from: {}'.format(layer, ', '.join(layers)))

        # Determine interpolation scheme based on layer
        if 'kaart' in self.layer:
            self.interpolation = 'catrom'
        else:
            self.interpolation = 'none'

        self._load_properties_by_scale()

        self.level = 1
        self.tiles = {}
        self.images = {}

    def _load_properties_by_scale(self):
        """
        Method that loads the matrix tile properties per zoom level
        """

        self.matrixproperties = {}

        resp = requests.get(self.url+'/tms/1.0.0/'+ self.layer + '/' +self.crs)
        soup = bs4.BeautifulSoup(resp.text, 'lxml')

        MatrixProps = namedtuple('matrixproperties', [
            'bottom',
            'left',
            'xstep',
            'ystep',
            'matrixwidth',
            'matrixheight',
            'tilewidth',
            'tileheight',
            'pixelsize'
        ])

        props = xmltodict.parse(str(soup))['html']['body']['tilemap']

        self.minx = float(props['boundingbox']['@minx'])
        self.miny = float(props['boundingbox']['@miny'])
        self.maxx = float(props['boundingbox']['@maxx'])
        self.maxy = float(props['boundingbox']['@maxy'])

        self.tileheight = int(props['tileformat']['@height'])
        self.tilewidth = int(props['tileformat']['@width'])

        for i, tileset in enumerate(props['tilesets']['tileset']):
            level = int(tileset['@order'])
            pixelsize = float(tileset['@units-per-pixel'])
        
            self.matrixproperties[level] = MatrixProps(
                self.miny,
                self.minx,
                self.tilewidth * pixelsize,
                self.tileheight * pixelsize,
                2**level,
                2**level,
                self.tilewidth,
                self.tileheight,
                pixelsize,
            )

        # Determine minimum ans maximum level
        levels = list(self.matrixproperties.keys())
        self.minlevel = min(levels)
        self.maxlevel = max(levels)

    
    def get_tile_indices(self):
        """
        Get the indices of the tiles by extent. The indices depend on the
        bounding box and the level
        """
        Indices = namedtuple('indices', ['xmin', 'xmax', 'ymin', 'ymax'])

        # Get matrix properties for the level
        mp = self.matrixproperties[self.level]

        # Determine the x indices
        xmin = int(max(0, (self.bbox[0] - mp.left) // mp.xstep))
        xmax = int(min(mp.matrixwidth, (self.bbox[2] - mp.left) // mp.xstep))

        # Determine the y indices
        ymin = int(max(0, (self.bbox[1] - mp.bottom) // mp.ystep))
        ymax = int(min(mp.matrixheight, (self.bbox[3] - mp.bottom) // mp.ystep))

        return Indices(xmin, xmax, ymin, ymax)

    def _calculate_level(self, mpp):
        """
        Calculate level based on a certain meter per pixel in the view.
        Correct for minlevel and maxlevel
        """
        level = (11 - (np.log2(mpp))).round().astype(int)
        level = max(self.minlevel, min(level, self.maxlevel))
        return level

    def update_tiles(self, bbox, mpp):
        """
        Function to update tiles based on bounding box and level
        """

        newlevel = self._calculate_level(mpp)
        if newlevel != self.level:
            # Clear the figure
            for i in list(self.tiles.keys()):
                row = self.tiles[i]
                for j in list(row.keys()):
                    self.images[i][j].remove()
                    del self.images[i][j]
                    del self.tiles[i][j]

        self.level = newlevel
        height = self.matrixproperties[self.level].matrixheight
        self.bbox = bbox

        # Construct tilematrixset name
        tmsname = '{}{}'.format(self.crs, ':' + str(self.level) if self.level is not None else '')

        # Get indices by bbox
        self.indices = self.get_tile_indices()

        # Get tiles based on indices
        for i in range(self.indices.xmin, self.indices.xmax + 1):
            # Add row if not present yet
            if i not in self.tiles.keys():
                self.tiles[i] = {}

            for j in range(self.indices.ymin, self.indices.ymax + 1):

                # Check if tile is already collected
                if j in self.tiles[i].keys():
                    continue

                # Collect the tile
                try:
                    f = self.tms.gettile(
                        i,
                        j,
                        self.level,
                        None,
                        self.layer,
                        self.crs,
                        None
                    )
                    # Read the tile
                    self.tiles[i][j] = plt.imread(BytesIO(f.read()))

                # In can occur that due to a bad connection tiles are not
                # loaded.  In that case, just pass the error and skip the tile.
                except:
                    pass

    def plot_and_clean(self, ax, clip):
        """
        This function plots tiles that have not been plotted yet and removes
        parts that are to far from the bounding box.
        """

        mp = self.matrixproperties[self.level]

        for i in list(self.tiles.keys()):
            row = self.tiles[i]
            # If the row is to far from the bounding box, remove.
            if i < (self.indices.xmin - 1) or i > (self.indices.xmax + 1):
                for j, im in self.images[i].items():
                    im.remove()
                del self.images[i]
                del self.tiles[i]
                continue

            # Add row if not present yet
            if i not in self.images.keys():
                self.images[i] = {}

            for j in list(row.keys()):
                tile = row[j]
                # If the tile is to far from the bounding box, remove.
                if j < (self.indices.ymin - 1) or j > (self.indices.ymax + 1):
                    self.images[i][j].remove()
                    del self.images[i][j]
                    del self.tiles[i][j]
                    continue

                # Check if tile is already plotted
                if j in self.images[i].keys():
                    continue

                # plot the image and add to the dictionary
                extent = (
                    mp.left + i * mp.xstep,
                    mp.left + (i + 1) * mp.xstep,
                    mp.bottom + j * mp.ystep,
                    mp.bottom + (j + 1) * mp.ystep,
                )

                im = ax.imshow(tile, extent=extent, interpolation=self.interpolation)
                self.images[i][j] = im

    def clean_all(self):

        for i in list(self.tiles.keys()):
            for j, im in self.images[i].items():
                im.remove()
            del self.images[i]
            del self.tiles[i]



class ListModel(QtCore.QAbstractListModel):
    def __init__(self, datain, parent=None, *args):
        """ datain: a list where each item is a row
        """
        super(ListModel, self).__init__(parent, *args)
        self.listdata = datain

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.listdata)

    def data(self, index, role):
        if index.isValid() and role == Qt.Qt.DisplayRole:
            return QtCore.QVariant(self.listdata[index.row()])
        else:
            return QtCore.QVariant()

        # -*- coding: utf-8 -*-
