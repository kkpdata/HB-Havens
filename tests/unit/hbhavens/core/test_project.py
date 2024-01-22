import unittest

from hbhavens import ui

class Test_project(unittest.TestCase):

    def test_user(self):
        p = ui.project.Project()

        p.set_user('Johan Ansink')

        self.assertEqual(p.get_user(), 'Johan Ansink')

    def test_email(self):
        p = ui.project.Project()

        p.set_email('j.ansink@hkv.nl')

        self.assertEqual(p.get_email(), 'j.ansink@hkv.nl')
        
    def test_save(self):
        p = ui.project.Project()
        p.set_user('Johan Ansink')
        p.set_email('j.ansink@hkv.nl')
        p.save()

if __name__ == '__main__':
    unittest.main()
