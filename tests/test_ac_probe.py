import json

from flybrain_drive.ac_probe import inspect_installation


def test_installed_engine_does_not_imply_runtime_capabilities(tmp_path):
    steam = tmp_path / "steam"
    library = tmp_path / "library"
    (steam / "steamapps").mkdir(parents=True)
    (steam / "steamapps/libraryfolders.vdf").write_text(
        '"libraryfolders" { "1" { "path" ' + json.dumps(str(library)) + ' } }'
    )
    (library / "steamapps").mkdir(parents=True)
    (library / "steamapps/appmanifest_244210.acf").write_text(
        '"AppState" { "appid" "244210" "installdir" "assettocorsa" }'
    )
    game = library / "steamapps/common/assettocorsa"
    game.mkdir(parents=True)
    (game / "acs.exe").write_bytes(b"fixture only, never executed")
    result = inspect_installation([steam])
    assert result["status"] == "engine_files_found"
    assert not result["game_launched"] and not result["actions_sent"]
    assert set(result["runtime_capabilities"].values()) == {"not_probed"}


def test_missing_install_and_manifest_traversal_are_reported(tmp_path):
    apps = tmp_path / "steamapps"
    apps.mkdir()
    (apps / "appmanifest_244210.acf").write_text('"installdir" "../../outside"')
    result = inspect_installation([tmp_path])
    assert result["status"] == "no_engine_in_inspected_locations"
    assert result["errors"][0]["error"] == "invalid_installdir"
    assert not result["candidates"]
