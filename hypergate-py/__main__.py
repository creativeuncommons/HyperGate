"""
HyperGate CLI Module Entrypoint
Allows running: python3 -m hypergate --list / --summary / --trace <id>
"""
from .audit import main

if __name__ == "__main__":
    main()
