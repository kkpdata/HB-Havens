import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from integration import simple_single_breakwater
from integration import simple_double_breakwater
from integration import simple_oosterschelde
from integration import simple_zoet_bovenrivieren
from integration import simple_zoet_benedenrivieren
from integration import simple_zoet_meren
from integration import simple_zoet_vijd
from integration import simple_zoet_europoort
from integration import simple_zoet_vzm
from integration import advanced_swan
from integration import advanced_swan_pharos
from integration import advanced_swan_hares
from integration import advanced_zoet
from integration import advanced_zoet_terugrekenen


# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(simple_single_breakwater))
suite.addTests(loader.loadTestsFromModule(simple_double_breakwater))
suite.addTests(loader.loadTestsFromModule(simple_oosterschelde))
suite.addTests(loader.loadTestsFromModule(simple_zoet_bovenrivieren))
suite.addTests(loader.loadTestsFromModule(simple_zoet_benedenrivieren))
suite.addTests(loader.loadTestsFromModule(simple_zoet_meren))
suite.addTests(loader.loadTestsFromModule(simple_zoet_vijd))
suite.addTests(loader.loadTestsFromModule(simple_zoet_europoort))
suite.addTests(loader.loadTestsFromModule(simple_zoet_vzm))
suite.addTests(loader.loadTestsFromModule(advanced_swan))
suite.addTests(loader.loadTestsFromModule(advanced_swan_pharos))
suite.addTests(loader.loadTestsFromModule(advanced_swan_hares))
suite.addTests(loader.loadTestsFromModule(advanced_zoet))
suite.addTests(loader.loadTestsFromModule(advanced_zoet_terugrekenen))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)