import argparse
import json
import os
import sys

if __name__ == '__main__':
    # python .\utils\interhands_frame_extractor.py --annot-path .\data\interhands\InterHand2.6M_test_MANO_NeuralAnnot.json --capture 1 --frame 24244 --hand right
    parser = argparse.ArgumentParser(description="Interhands extraction parameters")
    parser.add_argument("--annot-path", type=str, required=True, help="Path to interhands MANO annotations json")
    parser.add_argument("--capture", type=str, required=True, help="interhands capture")
    parser.add_argument("--frame", type=str, required=True, help="interhands frame")
    parser.add_argument("--hand", type=str, required=True, help="interhands hand: left or right")
    parser.add_argument("--save-path", type=str, required=True, help="where to save extracted frame for colmap")

    args, _ = parser.parse_known_args(sys.argv[1:])

    with open(args.annot_path,'r') as f_in:
        annotations = json.load(f_in)

    my_annot = annotations[args.capture][args.frame][args.hand]
    my_annot['is_rhand'] = True if args.hand=='right' else False
    tmp = {}
    count = {}
    for frame in sorted(annotations[args.capture].keys()):
        if annotations[args.capture][frame][args.hand] is None:
            continue
        # json circular reference error
        tmp[frame] = {k:v for k,v in annotations[args.capture][frame][args.hand].items()}
        print(sorted(annotations[args.capture][frame][args.hand].keys()))
        for k,v in annotations[args.capture][frame][args.hand].items():
            if k not in ['shape','pose']:
                continue
            if k not in count:
                count[k] = {}
            if len(v) not in count[k]:
                count[k][len(v)] = 0
            count[k][len(v)] += 1
    my_annot['all_frames'] = tmp
    print(my_annot.keys())
    print(count)
    with open(os.path.join(args.save_path,'mano_params.json'),'w') as f_out:
        json.dump(my_annot,f_out)

