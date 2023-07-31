import argparse

from main import GA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", help='Config file to use')
    parser.add_argument('--times', type=int, default="", help='run GA X times')
    args = parser.parse_args()
    print("Running GA with config: ", args.config, " times: ", args.times, " times")
    for i in range(0, int(args.times)):
        ga = GA(args.config)
        ga.run()
        print("Finished run: ", i, " of ", args.times, " times")