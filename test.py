#!/usr/bin/env python3
import sys
import site

def main():
    print("=== sys.executable ===")
    print(sys.executable, end="\n\n")

    print("=== sys.path entries ===")
    for p in sys.path:
        print(p)
    print()

    print("=== site-packages directories ===")
    for sp in site.getsitepackages():
        print(sp)

if __name__ == "__main__":
    main()
