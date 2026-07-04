
from leagents.doctor import (
    FAIL,
    OK,
    WARN,
    Check,
    check_headless_env,
    check_libero,
    check_llm_keys,
    check_python,
    run_doctor,
    run_setup,
)


def test_python_check_passes_here():
    assert check_python().level == OK


def test_headless_env_detects_missing_and_set():
    assert check_headless_env({}).level == WARN
    good = {"MUJOCO_GL": "egl", "SDL_VIDEODRIVER": "dummy"}
    assert check_headless_env(good).level == OK


def test_llm_keys_reports_providers_without_values():
    result = check_llm_keys({"GEMINI_API_KEY": "secret-value"})
    assert result.level == OK
    assert "gemini" in result.detail
    assert "secret-value" not in result.detail  # never leak values
    assert check_llm_keys({}).level == WARN


def test_libero_headers_missing_is_actionable(tmp_path):
    import leagents.doctor as doctor

    if doctor._has_module("libero"):
        assert check_libero().level == OK
    else:
        result = check_libero(include_dir=tmp_path / "no-such-EGL")
        assert result.level == FAIL
        assert "apt install" in (result.fix or "")


def test_run_doctor_exit_codes(capsys):
    ok = lambda: Check("a", OK, "fine")  # noqa: E731
    warn = lambda: Check("b", WARN, "meh", "do x")  # noqa: E731
    fail = lambda: Check("c", FAIL, "broken", "do y")  # noqa: E731
    assert run_doctor([ok, warn]) == 0
    assert run_doctor([ok, fail]) == 1
    out = capsys.readouterr().out
    assert "do y" in out and "1 failures" in out


def test_setup_creates_env_from_example(tmp_path, capsys):
    example = tmp_path / ".env.example"
    example.write_text("KEY=placeholder\n")
    target = tmp_path / ".env"
    assert run_setup(env_example=example, env_file=target) == 0
    assert target.read_text() == "KEY=placeholder\n"
    # never overwrite an existing .env
    target.write_text("KEY=real\n")
    run_setup(env_example=example, env_file=target)
    assert target.read_text() == "KEY=real\n"


def test_setup_reports_nothing_to_fix(tmp_path, capsys):
    assert run_setup(env_example=tmp_path / "none", env_file=tmp_path / ".env") == 0
    assert "nothing to fix" in capsys.readouterr().out
