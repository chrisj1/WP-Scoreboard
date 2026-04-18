# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('manifest.json', '.')],
    hiddenimports=[
        # Action modules — not reachable by static analysis (dynamic import)
        'src.actions.base_action',
        'src.actions.info_actions',
        'src.actions.score_actions',
        'src.actions.period_actions',
        'src.actions.exclusion_actions',
        'src.actions.timeout_actions',
        'src.actions.manup_actions',
        'src.actions.game_actions',
        'src.actions.replay_actions',
        # Pillow sub-modules
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        # websocket-client internals
        'websocket',
        'websocket._app',
        'websocket._core',
        'websocket._exceptions',
        'websocket._handshake',
        'websocket._http',
        'websocket._logging',
        'websocket._socket',
        'websocket._ssl_compat',
        'websocket._utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
