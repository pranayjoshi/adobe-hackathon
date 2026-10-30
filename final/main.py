"""Competition entry point: the starter's runner (engine.main) driving the final engine."""
import sys

import engine.main
from final.resize import resize

engine.main.resize = resize

if __name__ == '__main__':
    sys.exit(engine.main.main())
