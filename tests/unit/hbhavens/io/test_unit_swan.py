
import unittest
import env

import hbhavens.io

class unit_swan(unittest.TestCase):

    def unit_swan_test(self):
        print('test hello')
        fharborarea = os.path.join('vlissingen', 'haventerrein.shp')
        fbreakwater = os.path.join('vlissingen', 'havendammen.shp')
        fflooddefence = os.path.join('vlissingen', 'waterkeringlijn.shp')
        fdatabase = os.path.join('vlissingen', 'WBI2017_Westerschelde_29-3_29-4_v01.sqlite')

        supportlocation = 'WS_1_29-3_dk_00024'

        swan = core.swan.Swan(fharborarea, fbreakwater, fflooddefence, fdatabase, supportlocation)
        swan.run

if __name__ == '__main__':
    unittest.main()
