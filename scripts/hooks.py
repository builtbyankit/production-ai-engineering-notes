"""Publish the generated static site directly, without a Jekyll pass."""
from pathlib import Path


def on_post_build(config, **kwargs):
    (Path(config['site_dir']) / '.nojekyll').touch()
