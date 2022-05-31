import tensorflow as tf
from experiment.core import registry


class RegistryTest(tf.test.TestCase):

  def test_register(self):
    collection = {}

    @registry.register(collection, 'functions/func_0')
    def func_test():
      pass

    self.assertEqual(registry.lookup(collection, 'functions/func_0'), func_test)

    @registry.register(collection, 'classes/cls_0')
    class ClassRegistryKey:
      pass

    self.assertEqual(
        registry.lookup(collection, 'classes/cls_0'), ClassRegistryKey)

    @registry.register(collection, ClassRegistryKey)
    class ClassRegistryValue:
      pass

    self.assertEqual(
        registry.lookup(collection, ClassRegistryKey), ClassRegistryValue)

  def test_register_hierarchy(self):
    collection = {}

    @registry.register(collection, 'functions/func_0')
    def func_test0():
      pass

    @registry.register(collection, 'func_1')
    def func_test1():
      pass

    @registry.register(collection, func_test1)
    def func_test2():
      pass

    expected_collection = {
        'functions': {
            'func_0': func_test0,
        },
        'func_1': func_test1,
        func_test1: func_test2,
    }
    self.assertEqual(collection, expected_collection)

  def test_register_error(self):
    collection = {}

    @registry.register(collection, 'functions/func_0')
    def func_test0():  # pylint: disable=unused-variable
      pass

    with self.assertRaises(KeyError):

      @registry.register(collection, 'functions/func_0/sub_func')
      def func_test1():  # pylint: disable=unused-variable
        pass

    with self.assertRaises(LookupError):
      registry.lookup(collection, 'non-exist')


if __name__ == '__main__':
  tf.test.main()
