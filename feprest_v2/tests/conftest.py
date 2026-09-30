import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# real REST2 mdp templates, shared by the mdp tests
MDP_DIR = Path(__file__).resolve().parent.parent / "rundir_template" / "mdp"
