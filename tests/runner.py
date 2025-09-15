import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from unit.core.simple import LocalWaveGrowth

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

# initialize the TestLoader
loader = unittest.TestLoader()

# add tests to suiteUnit
suiteUnit = unittest.TestSuite()
suiteUnit.addTests(loader.loadTestsFromModule(LocalWaveGrowth))

# add tests to suiteIntegration
suiteIntegration = unittest.TestSuite()
suiteIntegration.addTests(loader.loadTestsFromModule(simple_single_breakwater))
suiteIntegration.addTests(loader.loadTestsFromModule(simple_double_breakwater))
suiteIntegration.addTests(loader.loadTestsFromModule(simple_oosterschelde))
suiteIntegration.addTests(loader.loadTestsFromModule(simple_zoet_bovenrivieren))
suiteIntegration.addTests(loader.loadTestsFromModule(simple_zoet_benedenrivieren))
suiteIntegration.addTests(loader.loadTestsFromModule(simple_zoet_meren))
suiteIntegration.addTests(loader.loadTestsFromModule(simple_zoet_vijd))
suiteIntegration.addTests(loader.loadTestsFromModule(simple_zoet_europoort))
suiteIntegration.addTests(loader.loadTestsFromModule(simple_zoet_vzm))
suiteIntegration.addTests(loader.loadTestsFromModule(advanced_swan))
suiteIntegration.addTests(loader.loadTestsFromModule(advanced_swan_pharos))
suiteIntegration.addTests(loader.loadTestsFromModule(advanced_swan_hares))
suiteIntegration.addTests(loader.loadTestsFromModule(advanced_zoet))
suiteIntegration.addTests(loader.loadTestsFromModule(advanced_zoet_terugrekenen))

# initialize a runner, pass it your suites and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suiteUnit)
result = runner.run(suiteIntegration)