from htcondor.htchirp import HTChirp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-msg',
                        type=str,
                        default='Hello World',
                        help='Message to send to job log')
    args = parser.parse_args()
    with HTChirp() as chirp:
        chirp.ulog(args.msg)

