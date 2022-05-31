"""Global streamz counters."""

from tensorflow.python.eager import monitoring


progressive_policy_creation_counter = monitoring.Counter(
    "/tensorflow/training/fast_training/progressive_policy_creation",
    "Counter for the number of ProgressivePolicy creations.")


stack_vars_to_vars_call_counter = monitoring.Counter(
    "/tensorflow/training/fast_training/tf_vars_to_vars",
    "Counter for the number of low-level stacking API calls.")
