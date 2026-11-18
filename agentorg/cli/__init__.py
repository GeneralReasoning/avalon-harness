"""Command-line tools: `avalon` (`sample.py`) runs organisations on OpenReward
tasks, `avalon-inspect` (`inspect_rollout.py`) reads a session directory, and
`cli_view.py` renders the live trace. The OpenReward sampler driver is under
`lib/`.

The harness core depends only on the `EnvAdapter` protocol
(`agentorg.env_adapter`); the OpenReward adapter it ships is
`agentorg.openreward_env`.
"""
