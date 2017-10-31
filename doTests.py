# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 07:54:41 2017

@author: hseltman
"""

import unittest


class TestStringMethods(unittest.TestCase):
    """ class to test upper(), isupper(), and split() """
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
