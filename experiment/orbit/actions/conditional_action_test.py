from experiment.orbit import actions
import tensorflow as tf


class ConditionalActionTest(tf.test.TestCase):

  def test_conditional_action(self):
    # Define a function to raise an AssertionError, since we can't in a lambda.
    def raise_assertion(arg):
      raise AssertionError(str(arg))

    conditional_action = actions.ConditionalAction(
      condition=lambda x: x['value'], action=raise_assertion)

    conditional_action({'value': False}) # Noting is raised.
    with self.assertRaises(AssertionError) as ctx:
      conditional_action({'value': True})
      self.assertEqual(ctx.exception.message, '{\'value\': True}')

if __name__ == '__main__':
  tf.test.main()
