# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller build spec for the offline-first, one-file .exe.
#
# Build:  pyinstaller recon-agent.spec
# Output: dist/recon-agent.exe
#
# Verified end-to-end (not just built): ran the resulting .exe from a
# clean directory away from the source tree, confirmed /health, the
# dashboard, and /datasets all work, confirmed no synthetic data is
# present (production mode - see src/launcher.py), and confirmed an
# uploaded file persists to <exe's directory>/data/uploaded/ - NOT
# PyInstaller's temp extraction folder, which is wiped on every launch
# and would otherwise silently lose the user's data between sessions.
# See src/paths.py's module docstring for why that distinction matters
# and how it's enforced.
#
# console=True (see EXE() below) deliberately: startup errors are visible
# in the console window rather than silently swallowed, which matters more
# for a submission/demo build than a fully chrome-less window. Flip to
# False for a more polished release build once you're confident nothing
# needs debugging.

from PyInstaller.utils.hooks import collect_all

datas = [('static', 'static')]
binaries = []
hiddenimports = [
    'uvicorn.logging',
    'uvicorn.lifespan.on',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.loops.auto',
]

# collect_all pulls in each package's full data/binary/hidden-import set —
# needed because PyInstaller's static import analysis misses uvicorn's
# protocol/loop plugins (loaded dynamically by name) and pywebview's
# platform backend (Edge WebView2 on Windows, chosen at runtime).
for pkg in ('uvicorn', 'fastapi', 'webview'):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    ['src/launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='recon-agent',
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
