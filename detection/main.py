import argparse
import importlib
import pathlib
import sys
import logging
import os

VERSION = "0.0.0"
PACKAGE_NAME = os.path.dirname(__file__).split("/")[-1]
CLI_PROG = "detection"
__version__ = VERSION
__packaging_server_version__ = "nota_package_server@v1.4.0"

def command_run(args):
    is_dir = False
    is_file = False
    import cv2
    if os.path.isdir(args.input):
        is_dir = True
    elif os.path.isfile(args.input):
        is_file = True
    else:
        raise Exception
    
    model = importlib.import_module(PACKAGE_NAME+".models.model")
    base = importlib.import_module(PACKAGE_NAME+".models.base")
    
    for i in [models_model_callable_obj 
                for models_model_callable_obj in dir(model) 
                    if callable(getattr(model, models_model_callable_obj))]:
        try:
            if base.Basemodel in getattr(model, i).__bases__: 
                models_main_class = getattr(model, i)
        except:
            continue
    
    models_main_class.initialize(num_threads=args.num_threads)
    models = models_main_class()
    if is_dir:
        for img_file in os.listdir(args.input):
            if args.raw_input:
                image = cv2.imread(f"{args.input}/{img_file}")
                if image is None:
                    raise Exception()
            else:
                image = f"{args.input}/{img_file}"
            models.run(image, 
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres)
    elif is_file:
        image = cv2.imread(str(args.input))
        models.run(image, 
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres)
        if image is None:
            raise Exception()
    models_main_class.finalize()

def add_options(parser):
    parser.add_argument('-v', '--version', action='version',
                        version=f'{PACKAGE_NAME} {__version__} '
                                f'Packaging by {__packaging_server_version__}')
    parser.add_argument('-q', '--silent', action='store_true',
                        help='Suppress all normal output')

def _parser():
    parser = argparse.ArgumentParser(
        prog=CLI_PROG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=f'See "{CLI_PROG} <commnd> -h" for more information '
                'on a specific command. \n\nMore usage refer to'
            #    f' {DOC_URL}'
    )
    add_options(parser)

    subparsers = parser.add_subparsers(
        title=f'The most commonly used {CLI_PROG} commands are',
        metavar='COMMAND'
    )

    #
    # Command: run
    #
    cparser = subparsers.add_parser(
        'run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Run model inference'
    )
    cparser.add_argument('input', 
                            type=pathlib.Path, metavar="INPUT_PATH",
                            help="input files or folder"),
    cparser.add_argument('--raw_input', 
                            required=False,
                            default=False, action='store_true',
                            help="Check the input file is raw (default: False)")
    cparser.add_argument('--num_threads', 
                            required=False, metavar="",
                            type=int, default=1,
                            help="Number of threads (default: 1)")
    cparser.add_argument('--conf_thres', 
                            required=False, metavar="",
                            type=float, default=0.25,
                            help="Value of Confidence Threshold  (default: 0.25)")
    cparser.add_argument('--iou_thres', 
                            required=False, metavar="",
                            type=float, default=0.60,
                            help="Value of IoU Threshold (default: 0.60)")
    cparser.set_defaults(func=command_run)

    return parser

def main(argv):
    parser = _parser()
    args = parser.parse_args(argv)
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    if args.silent:
        logging.getLogger().setLevel(999)
    
    args.func(args)

def entry_point():
    main(sys.argv[1:])

if __name__ == '__main__':
    entry_point()