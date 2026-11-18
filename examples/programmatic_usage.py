#!/usr/bin/env python
"""Programmatic usage of Avalon Harness.

This example shows the library API without going through the CLI: building
your own `Harness`, configuring a session, running it to completion, and
printing the reward and where the session was written on disk.

It runs against `GeneralReasoning/counter` task 0 with budget=0 (solo CEO),
using the mode (`org-plan`) and compaction settings Avalon-M1-Mini was
trained with. Temperature 1.0, as in training, is the `SessionConfig` default.
"""

from __future__ import annotations

import asyncio
import os

from openreward import AsyncOpenReward

from agentorg.compaction import avalon_compaction_config
from agentorg.openreward_env import OpenRewardEnvAdapter
from agentorg.harness import Harness
from agentorg.session import SessionConfig
from agentorg.storage import session_dir
from agentorg.types import EnvironmentBinding


async def main() -> None:
    harness = Harness(
        base_url=os.environ.get("AVALON_BASE_URL", "http://localhost:30000/v1"),
        api_key=os.environ.get("AVALON_API_KEY", "none"),
    )

    client = AsyncOpenReward(api_key=os.environ["OPENREWARD_API_KEY"])
    env = client.environments.get("GeneralReasoning/counter")
    task = await env.get_task("train", 0)

    async with env.session(task=task) as or_session:
        adapter = OpenRewardEnvAdapter(
            client=client,
            session=or_session,
            deployment_name="GeneralReasoning/counter",
            split="train",
            task_index=0,
        )
        blocks = await or_session.get_prompt()
        preamble = "\n\n".join(b.text for b in blocks if getattr(b, "type", None) == "text")

        config = SessionConfig(
            workspace_path="./rollouts/example",
            model="GeneralReasoning/Avalon-M1-Mini",
            budget=0,
            mode="org-plan",
            environment=EnvironmentBinding(
                kind="openreward",
                task_id="GeneralReasoning/counter",
                split="train",
            ),
            headless=True,
            permission_policy="bypass",
            initial_user_message=preamble or None,
            compaction=avalon_compaction_config(),
        )
        session = await harness.start_session(config=config, env_adapter=adapter)
        result = await harness.run_to_completion(session, max_wall_s=1800)

    print(f"Session: {result.session_id}")
    print(f"  reward       = {result.reward}")
    print(f"  duration_s   = {result.duration_s:.2f}")
    print(f"  used_turns   = {result.used_turns}")
    print(f"  org_agents   = {[a.name for a in result.org_agents]}")
    print(f"  session_dir  = {session_dir(result.workspace_path, result.session_id)}")

    await client.environments.aclose()


if __name__ == "__main__":
    asyncio.run(main())
