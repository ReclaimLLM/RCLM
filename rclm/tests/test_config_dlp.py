from rclm import _config


def test_dlp_defaults_on_when_setting_is_missing():
    assert _config.dlp_enabled({}) is True


def test_explicit_dlp_opt_out_is_preserved():
    assert _config.dlp_enabled({"dlp": False}) is False
