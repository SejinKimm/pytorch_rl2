"""
Implements common agent components used in Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

import torch as tc


class WeightNormedLinear(tc.nn.Module):
    """A linear layer with weight normalization included,
    and support for optional initialization of non-unit gains.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        gain_init=1.0,
        weight_initializer=None,
        bias_initializer=None
    ):
        super().__init__()
        self._linear = tc.nn.Linear(input_dim, output_dim)
        if weight_initializer is not None:
            weight_initializer(self._linear.weight)
        if bias_initializer is not None:
            bias_initializer(self._linear.bias)
        self.linear = tc.nn.utils.weight_norm(self._linear, name='weight')
        with tc.no_grad():
            self.linear.weight_g.copy_(gain_init * tc.ones(output_dim, 1))

    def forward(self, x):  # pylint: disable=C0116
        return self.linear(x)


class PolicyHead(tc.nn.Module):
    """
    Policy head for a reinforcement learning agent.
    Uses a weight-normed linear layer.
    """
    def __init__(self, feature_dim, num_actions):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.linear = WeightNormedLinear(
            input_dim=self.feature_dim,
            output_dim=self.num_actions,
            gain_init=0.01)

    def forward(self, features: tc.FloatTensor) -> tc.distributions.Categorical:
        """
        Compute a policy distributions from features and return it.

        Args:
            features: a tc.FloatTensor of features of shape [B, feature_dim].

        Returns:
            a tc.distributions.Categorical over actions, with batch shape [B].
        """
        logits = self.linear(features)
        dist = tc.distributions.Categorical(logits=logits)
        return dist


class ValueHead(tc.nn.Module):
    """
    Value head for a reinforcement learning agent.
    Uses a weight-normed linear layer.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.linear = WeightNormedLinear(
            input_dim=self.feature_dim,
            output_dim=1,
            gain_init=1.0)

    def forward(self, features: tc.FloatTensor) -> tc.FloatTensor:
        """
        Compute a policy distributions from features and return it.

        Args:
            features: a tc.FloatTensor of features with shape [B, feature_dim].

        Returns:
            a tc.FloatTensor of value estimates with shape [B].
        """
        v_pred = self.linear(features).squeeze(-1)
        return v_pred


def one_hot(ys: tc.LongTensor, depth: int) -> tc.FloatTensor:
    """
    Applies one-hot encoding to a batch of vectors.

    Args:
        ys: tc.LongTensor of shape [B].
        depth: int specifying the number of possible label values.

    Returns:
        one-hot encodings of ys.
    """

    batch_size = ys.shape[0]
    vecs_shape = [batch_size, depth]
    vecs = tc.zeros(dtype=tc.float32, size=vecs_shape)
    vecs.scatter_(dim=1, index=ys.unsqueeze(-1),
                  src=tc.ones(dtype=tc.float32, size=vecs_shape))
    return vecs

