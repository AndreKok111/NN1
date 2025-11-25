"""
Tests for the main application
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import main


def test_main():
    """Test main function"""
    # This is a placeholder test
    assert callable(main)


if __name__ == "__main__":
    test_main()
    print("Tests passed!")

