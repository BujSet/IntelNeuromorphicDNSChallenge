from htchirp_utils import send_log_msg
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-msg',
                        type=str,
                        default='Hello world',
                        help='Message to append to CHTC log files')
    args = parser.parse_args()
    send_log_msg(args.msg)
