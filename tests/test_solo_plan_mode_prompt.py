from agentorg.openreward_env import _looks_read_only
from agentorg.prompts import build_ceo_system_prompt


def _solo_plan_prompt() -> str:
    return build_ceo_system_prompt(
        mode="org-plan", budget=0, env=None, toolset_name=None, live_org_agents=[],
        has_filesystem_tools=False, headless=True, workspace_agents_md=None,
        date_str=None,
    )


def test_solo_plan_prompt_no_longer_says_to_use_every_env_tool():
    prompt = _solo_plan_prompt()
    assert "use the environment tools in your toolset" not in prompt
    assert "no local filesystem tools" not in prompt
    assert "Until you call ExitOrgPlanMode" in prompt


def test_tools_the_prompt_names_match_the_plan_mode_gate():
    prompt = _solo_plan_prompt()
    for allowed in ("ls", "read", "view", "list", "get", "grep", "glob", "find", "search"):
        assert allowed in prompt
        assert _looks_read_only(allowed), allowed
    for denied in ("bash", "write", "edit"):
        assert denied in prompt
        assert not _looks_read_only(denied), denied
