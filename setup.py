"""
Setup script for sid-unet with automated installation-time 8-bit compatibility checking.
Metadata is driven by pyproject.toml; this setup file hooks into the build/develop
lifecycle to automatically verify CUDA, GPU architecture, and 8-bit training readiness.
"""

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


class CustomBuildPy(build_py):
    """Custom build_py hook running automated 8-bit compatibility verification."""

    def run(self):
        try:
            from sid_unet.utils.compatibility import installation_check_8bit
            installation_check_8bit()
        except Exception:
            pass
        super().run()


class CustomDevelop(develop):
    """Custom develop hook running automated 8-bit compatibility verification for editable installs."""

    def run(self):
        try:
            from sid_unet.utils.compatibility import installation_check_8bit
            installation_check_8bit()
        except Exception:
            pass
        super().run()


if __name__ == "__main__":
    setup(
        cmdclass={
            "build_py": CustomBuildPy,
            "develop": CustomDevelop,
        }
    )
