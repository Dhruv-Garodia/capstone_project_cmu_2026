"""Where the generator looks for things outside the repo. Each is an environment variable;
nothing here is needed just to generate stacks.

PFIB_REAL_DIR    the real stacks comp_full.tif / uncomp_full.tif / pristine_full.tif (realism scoring, recalibration)
PFIB_LABELS_DIR  the team's labelled review packages team_review, team_review_uncomp, team_review_pristine
PFIB_OUT         where packages are written and read back for evaluation (default: <repo>/out)
PFIB_TEAM_REPO   a checkout of the team repo, for its U-Net (train_shine.py)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REAL_DIR = Path(os.environ.get("PFIB_REAL_DIR", REPO / "data" / "real"))
LABELS_DIR = Path(os.environ.get("PFIB_LABELS_DIR", REPO / "data" / "labels"))
OUT = Path(os.environ.get("PFIB_OUT", REPO / "out"))
TEAM_REPO = Path(os.environ.get("PFIB_TEAM_REPO", REPO.parent / "capstone_project_cmu_2026"))
