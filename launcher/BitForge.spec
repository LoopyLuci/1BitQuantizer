# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os
from pathlib import Path

ROOT = Path('.').resolve()
FRONTEND_DIST = ROOT.parent / 'tauri-gui' / 'dist'
BACKEND_SERVICE = ROOT.parent / 'backend_service.py'
ENTRY = ROOT / 'app.py'

datas = []
if FRONTEND_DIST.exists():
    datas.append((str(FRONTEND_DIST), 'tauri-gui/dist'))
if BACKEND_SERVICE.exists():
    datas.append((str(BACKEND_SERVICE), '.'))

a = Analysis(
    [str(ENTRY)],
    pathex=[str(ROOT / 'launcher'), str(ROOT / 'backend'), str(ROOT.parent)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'webview',
        'webview.platforms.edgechromium',
        'uvicorn',
        'backend_service',
        'appdirs',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='BitForge',
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(ROOT.parent / 'tauri-gui' / 'src-tauri' / 'icons' / 'icon.ico'),
)
