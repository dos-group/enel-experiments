from unittest import TestCase

from multi_class_data_gen import MultiClassDataGenerator


class TestMultiClassDataGenerator(TestCase):
    def test_data(self):
        generator = MultiClassDataGenerator(2000, 4, 3)
        data = generator.generate_data("test.txt")
        print(data)
