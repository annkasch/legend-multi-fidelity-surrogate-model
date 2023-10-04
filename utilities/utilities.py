#!/usr/bin/env python3

import sys
import argparse
from termcolor import colored


def INFO(output):
    try:
        print(colored('[INFO] '+output, 'green'))
    except:
        print(colored('[INFO] '+str(output), 'green'))

def WARN(output):
    try:
        print(colored('[WARNING] '+output, 'yellow'))
    except:
        print(colored('[WARNING] '+str(output), 'yellow'))

def ERROR(output):
    try:
        print(colored('[ERROR] '+output, 'red'))
    except:
        print(colored('[ERROR] '+str(output), 'red'))
    sys.exit()

