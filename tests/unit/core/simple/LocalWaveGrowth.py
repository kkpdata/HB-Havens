import unittest
import logging
import os
# import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
import geopandas as gpd
# import shapely

import common
import hbhavens
from hbhavens.core.logger import initialize_logger
from hbhavens.core import datamodels
from hbhavens.core.simple import LocalWaveGrowth


datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Bretschneider')
trunkdir = os.path.abspath(os.path.join(os.path.dirname(hbhavens.__file__),"../.."))
tempdir = os.path.join(trunkdir, 'temp')

paths = {
    'temp': os.path.join(tempdir),
    'harborarea': os.path.join(datadir, 'shapes', 'haventerrein.shp'),
    'breakwater': os.path.join(datadir, 'shapes', 'havendammen.shp'),
    'result_locations': os.path.join(datadir, 'shapes', 'uitvoerlocaties.shp'),
    'hrd': os.path.join(datadir, 'databases', 'WBI2017_Vlissingen_steunpunt.sqlite'),
    'config': os.path.join(datadir, 'databases', 'WBI2017_Vlissingen_steunpunt.config.sqlite'),
    'hlcd': os.path.join(datadir, '..', 'hlcd.sqlite'),
}

initialize_logger(filename=os.path.join(tempdir,'hbhavens.log'))
logger = logging.getLogger()

class Test_LocalWaveGrowth_1(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(paths['temp']):
            os.mkdir(paths['temp'])
        logger.info('SETUP Unit - Module core.simple - Class LocalWaveGrowth')

    def tearDown(self):
        logger.info('TEARDOWN Unit - Module core.simple - Class LocalWaveGrowth')

    def test_determine_fetch_lines_1(self):
        '''
        Test module core.simple, class LocalWaveGrowth
        Test method _determine_fetch_lines

        Testing the fetches for a square box with sizes 1. and a point in the lower left corner (origin).
        self.step(size) = 1. (default)
        '''
        logger.info('test_determine_fetch_lines_1')

        # self verwijst niet naar een instantie LocalWaveGrowth omdat de initialisatie allerlei meta data vereist die niet nodig is voor unit testen.
        # Een praktische aanpak is om de strikt noodzakelijke attributen aan de self instantie (Test_LocalWaveGrowth_1) toe te voegen en de resultaten daarin te verzamelen
        # In de toekomst is een separate module gewenst die onafhankelijk van de vereisten van HB Havens de windgroei berekent en separaat getest kan worden.
        
        self.step = 1.
        pts = {0:{
                    'geometry': Point(0,0),
                    'Naam': 'ResultPoint1',
                    'Normaal': None
                }}
        self.result_locations = gpd.GeoDataFrame.from_dict(pts, orient='index')
        self.inner = Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))

        LocalWaveGrowth._determine_fetch_lines(self)

        self.assertEqual(self.fetchlines['fetch'].max(), np.sqrt(2.))
        self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',0.)].values[0], 1.)
        self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',45.)].values[0], np.sqrt(2))
        self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',90.)].values[0], 1.)
        self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',180.)].values[0], 0.)
        self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',270.)].values[0], 0.)

    def test_determine_fetch_lines_2(self):
        '''
        Test module core.simple, class LocalWaveGrowth
        Test method _determine_fetch_lines

        Testing the fetches for a circle with radius R=10000. and a point in the center (origin)
        self.step(size) = 1.
        '''

        logger.info('test_determine_fetch_lines_2')
        # self verwijst niet naar een instantie LocalWaveGrowth omdat de initialisatie allerlei meta data vereist die niet nodig is voor unit testen.
        # Een praktische aanpak is om de strikt noodzakelijke attributen aan de self instantie (Test_LocalWaveGrowth_1) toe te voegen en de resultaten daarin te verzamelen
        # In de toekomst is een separate module gewenst die onafhankelijk van de vereisten van HB Havens de windgroei berekent en separaat getest kan worden.
        R = 10000.
        self.step = 1.
        pts = {0:{
                    'geometry': Point(0,0),
                    'Naam': 'ResultPoint1',
                    'Normaal': 0.
                }}
        self.result_locations = gpd.GeoDataFrame.from_dict(pts, orient='index')

        # nauwkeurigheid wordt gedicteerd door de buffer in Shapely en die is laag (delta=1e-2)
        # # self.inner = Point(0, 0).buffer(R)
        
        directions = np.arange(0., 2*np.pi, 1e-4)
        x = R*np.sin(directions)
        y = R*np.cos(directions)
        self.inner = Polygon(zip(x,y))

        LocalWaveGrowth._determine_fetch_lines(self)

        self.assertEqual(self.fetchlines['fetch'].max(), R)
        for row in self.fetchlines.iterrows():
            self.assertAlmostEqual(row[1]['fetch'], R, None, None, delta=1e-4) 

    def test_determine_fetch_lines_3(self):
        '''
        Test module core.simple, class LocalWaveGrowth
        Test method _determine_fetch_lines

        Testing the fetches for a square box with sizes 1. and a point in the lower left corner (origin).
        self.step(size) = 10.
        '''

        logger.info('test_determine_fetch_lines_3')
        # self verwijst niet naar een instantie LocalWaveGrowth omdat de initialisatie allerlei meta data vereist die niet nodig is voor unit testen.
        # Een praktische aanpak is om de strikt noodzakelijke attributen aan de self instantie (Test_LocalWaveGrowth_1) toe te voegen en de resultaten daarin te verzamelen
        # In de toekomst is een separate module gewenst die onafhankelijk van de vereisten van HB Havens de windgroei berekent en separaat getest kan worden.
        
        self.step = 1.
        pts = {0:{
                    'geometry': Point(0,0),
                    'Naam': 'ResultPoint1',
                    'Normaal': None
                }}
        self.result_locations = gpd.GeoDataFrame.from_dict(pts, orient='index')
        self.inner = Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))

        LocalWaveGrowth._determine_fetch_lines(self)

        self.assertEqual(self.fetchlines['fetch'].max(), np.sqrt(2))
        self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',0.)].values[0], 1.)
        # self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',45.)].values[0], np.sqrt(2))
        self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',90.)].values[0], 1.)
        self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',180.)].values[0], 0.)
        self.assertEqual(self.fetchlines['fetch'].loc[self.fetchlines.index == ('ResultPoint1',270.)].values[0], 0.)

    def test_find_fetch_per_direction_1(self):
        '''
        Test module core.simple, class LocalWaveGrowth
        Test method _find_fetch_per_direction

        Testing the fetches for a square box with sizes 1. and a point in the lower left corner (origin).
        self.step(size) = 1. (default)
        self.spread_angle = 60. (default)
        '''

        logger.info('test_find_fetch_per_direction_1')
        self.spread_angle = 60. 
        self.step = 1.
        pts = {0:{
                    'geometry': Point(0,0),
                    'Naam': 'ResultPoint1',
                    'Normaal': None
                }}
        self.result_locations = gpd.GeoDataFrame.from_dict(pts, orient='index')
        self.inner = Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))

        LocalWaveGrowth._determine_fetch_lines(self)
        LocalWaveGrowth._find_fetch_per_direction(self,method='max')

        self.assertEqual(self.fetchlines['equi_fetch_direction'].loc[self.fetchlines.index == ('ResultPoint1',0.)].values[0], 30.)
        self.assertEqual(self.fetchlines['equi_fetch_direction'].loc[self.fetchlines.index == ('ResultPoint1',45.)].values[0], 45.)
        self.assertEqual(self.fetchlines['equi_fetch_direction'].loc[self.fetchlines.index == ('ResultPoint1',90.)].values[0], 60.)
        self.assertEqual(self.fetchlines['equi_fetch_direction'].loc[self.fetchlines.index == ('ResultPoint1',120.)].values[0], 90.)
        self.assertEqual(self.fetchlines['equi_fetch_direction'].loc[self.fetchlines.index == ('ResultPoint1',270.)].values[0], 240.)
        self.assertEqual(self.fetchlines['equi_fetch_direction'].loc[self.fetchlines.index == ('ResultPoint1',329.)].values[0], 299.)
        self.assertEqual(self.fetchlines['equi_fetch_direction'].loc[self.fetchlines.index == ('ResultPoint1',330.)].values[0], 0.)
        self.assertEqual(self.fetchlines['equi_fetch_direction'].loc[self.fetchlines.index == ('ResultPoint1',359.)].values[0], 29.)

    def test_calc_local_wave_growth_1(self):
        '''
        Test module core.simple, class LocalWaveGrowth
        Test method _calc_local_wave_growth

        Testing the evaluation of various tabulated pre-calculated combinations of fetch and wind speed.
        See Golfbelastingen in havens en afgeschermde gebieden (RWS, 2014), Table 2
        '''
        logger.info('test_calc_local_wave_growth_1')
        Feq,U10 = np.meshgrid(np.arange(100,2000+1e-6,100),np.arange(5,50+1e-6,5))
        Hs_lg = LocalWaveGrowth._calc_local_wave_growth(self,Feq,U10)

        self.assertAlmostEqual(Hs_lg.min(), 0.04, 2)
        self.assertAlmostEqual(Hs_lg.max(), 1.69, 2)
        self.assertAlmostEqual(Hs_lg[(Feq == 500. ) & ( U10 == 20)][0], 0.34, 2)
        self.assertAlmostEqual(Hs_lg[(Feq == 1000.) & ( U10 == 40)][0], 0.96, 2)
        self.assertAlmostEqual(Hs_lg[(Feq == 1800.) & ( U10 == 45)][0], 1.44, 2)

    def test_bretschneider_1(self):
        '''
        Test module core.simple, class LocalWaveGrowth
        Test method bretschneider

        Testing the evaluation of various combinations of depth, fetch and wind speed.
        The comparisons are made to outcomes of the Bretschneider calculator that is part of HydraNL
        '''
        logger.info('test_bretschneider_1')
        fe,u,d = np.meshgrid(np.arange(100,2000+1e-6,100),np.arange(5,50+1e-6,5),[1.,10,100,1000,10000])
        hs,tp = LocalWaveGrowth.bretschneider(self,d,fe,u)

        # test max/min outcomes
        self.assertAlmostEqual(hs.min(), 0.04, 2) # d=1,fe=100,u=5
        self.assertAlmostEqual(tp.min(), 0.77, 2)
        self.assertAlmostEqual(hs.max(), 2.14, 2) # d=10000,fe=2000,u=50
        self.assertAlmostEqual(tp.max(), 5.32, 2)

        # test random specific outcomes
        self.assertAlmostEqual(hs[(fe == 500. ) & (u == 20) & (d == 1000)][0], 0.41, 2)
        self.assertAlmostEqual(hs[(fe == 1000.) & (u == 40) & (d == 100)][0], 1.23, 2)
        self.assertAlmostEqual(hs[(fe == 1800.) & (u == 45) & (d == 1)][0], 0.57, 2)
        self.assertAlmostEqual(tp[(fe == 500. ) & (u == 20) & (d == 1000)][0], 2.38, 2)
        self.assertAlmostEqual(tp[(fe == 1000.) & (u == 40) & (d == 100)][0], 3.97, 2)
        self.assertAlmostEqual(tp[(fe == 1800.) & (u == 45) & (d == 1)][0], 3.47, 2)
        
    def test_bretschneider_2(self):
        '''
        Test module core.simple, class LocalWaveGrowth
        Test method bretschneider

        Testing input formats
        '''
        logger.info('test_bretschneider_2')
        d=1.
        fe=100.
        u=5.

        # test input types
        hs,tp = LocalWaveGrowth.bretschneider(self,int(d),int(fe),int(u))
        # self.assertAlmostEqual(hs[0], 0.04, 2) # ongelijk omdat de afronding tot andere invoer leidt en dus tot andere resultaten
        # self.assertAlmostEqual(tp[0], 0.77, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,float(d),float(fe),float(u))
        self.assertAlmostEqual(hs[0], 0.04, 2) 
        self.assertAlmostEqual(tp[0], 0.77, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,np.array(int(d)),np.array(int(fe)),np.array(int(u)))
        # self.assertAlmostEqual(hs.item(), 0.04, 2) # ongelijk omdat de afronding tot andere invoer leidt en dus tot andere resultaten
        # self.assertAlmostEqual(tp.item(), 0.77, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,np.array(float(d)),np.array(float(fe)),np.array(float(u)))
        self.assertAlmostEqual(hs.item(), 0.04, 2) 
        self.assertAlmostEqual(tp.item(), 0.77, 2)

        d=1000.
        fe=500.
        u=20.

        # test input types
        hs,tp = LocalWaveGrowth.bretschneider(self,int(d),int(fe),int(u))
        # self.assertAlmostEqual(hs[0], 0.41, 2) 
        # self.assertAlmostEqual(tp[0], 2.38, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,float(d),float(fe),float(u))
        self.assertAlmostEqual(hs[0], 0.41, 2) 
        self.assertAlmostEqual(tp[0], 2.38, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,np.array(int(d)),np.array(int(fe)),np.array(int(u)))
        # self.assertAlmostEqual(hs.item(), 0.41, 2) 
        # self.assertAlmostEqual(tp.item(), 2.38, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,np.array(float(d)),np.array(float(fe)),np.array(float(u)))
        self.assertAlmostEqual(hs.item(), 0.41, 2) 
        self.assertAlmostEqual(tp.item(), 2.38, 2)
        
    def test_bretschneider_3(self):
        '''
        Test module core.simple, class LocalWaveGrowth
        Test method bretschneider

        Testing unlikely input
        '''
        d=1000.
        fe=500.
        u=20.

        logger.info('test_bretschneider_3')
        
        # test input types
        hs,tp = LocalWaveGrowth.bretschneider(self,d,fe,u)
        self.assertAlmostEqual(hs[0], 0.41, 2) 
        self.assertAlmostEqual(tp[0], 2.38, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,1e-999,fe,u)
        self.assertAlmostEqual(hs.item(), 0.0, 2) 
        self.assertAlmostEqual(tp.item(), 0.0, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,1e999,fe,u)
        self.assertAlmostEqual(hs[0], 0.41, 2) 
        self.assertAlmostEqual(tp[0], 2.38, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,d,1e-999,u)
        self.assertAlmostEqual(hs.item(), 0.0, 2) 
        self.assertAlmostEqual(tp.item(), 0.0, 2)
        
        hs,tp = LocalWaveGrowth.bretschneider(self,d,1e999,u) # runs but Bretschneider calculator fails, so hard to compare
        # self.assertAlmostEqual(hs[0], 0.41, 2) 
        # self.assertAlmostEqual(tp[0], 2.38, 2)

        hs,tp = LocalWaveGrowth.bretschneider(self,d,fe,1e-999)
        self.assertAlmostEqual(hs.item(), 0.0, 2) 
        self.assertAlmostEqual(tp.item(), 0.0, 2)
        
        # hs,tp = LocalWaveGrowth.bretschneider(self,d,fe,1e999) # fails due to division by zero but Bretschneider calculator fails
        hs,tp = LocalWaveGrowth.bretschneider(self,d,fe,1e-99) 
        # self.assertAlmostEqual(hs[0], 0.41, 2) 
        # self.assertAlmostEqual(tp[0], 2.38, 2)

if __name__ == '__main__':
    unittest.main()
