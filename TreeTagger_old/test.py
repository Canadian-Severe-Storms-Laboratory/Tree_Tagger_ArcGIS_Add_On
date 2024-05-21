import sys
import time


def spinning_cursor():
    while True:
        for cursor in '|/—\\':
            yield cursor


if __name__ == '__main__':

    spinner = spinning_cursor()
    while True:
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        time.sleep(0.25)
        sys.stdout.write('\b')

