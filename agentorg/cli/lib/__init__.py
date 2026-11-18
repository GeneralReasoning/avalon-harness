"""OpenReward sampler driver: parallel rollouts of one environment (§12.2).

The harness depends on the `EnvAdapter` protocol (`agentorg.env_adapter`),
which `agentorg.openreward_env.OpenRewardEnvAdapter` implements; this package
only drives sessions through it.
"""
