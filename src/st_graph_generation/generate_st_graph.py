from os import getcwd, listdir
from os.path import join, isfile, isdir, exists, abspath, dirname, basename
from typing import Generator

def get_frame() -> Generator[str, None, None]:
    """Generator of all the frames in the current directory.
    """
    for file in listdir(getcwd()):
        if isfile(file) and file.endswith('.png'):
            yield file

def main() -> None:



if __name__ == "__main__":
    main()