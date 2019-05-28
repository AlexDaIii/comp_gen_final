import os
import subprocess
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--preprocess_gcp", action='store_true', help="Adding this argument will pre-process gcp data")
    parser.add_argument("--preprocess_gene", action='store_true', help="Adding this argument will pre-process gene data")
    parser.add_argument("--run_histone", action='store_true', help="Adding this argument will run elastic net and analysis on histone data")
    parser.add_argument("--run_gene", action='store_true', help="Adding this argument will run elastic net and analysis on gene data")
    parser.add_argument("--run_pca", action='store_true', help="Adding this argument will run pca on histone data")

    # preprocessing gcp
    parser.add_argument("--gcp_X", type=str, required=False, help="The unprocessed gcp data.")
    parser.add_argument("--gcp_Y", type=str, required=False, help="The unprocessed gcp labels.")
    parser.add_argument('--gcp_outputY', type=str, required=False, help='csv file to contain labels')
    parser.add_argument('--gcp_outputX', type=str, required=False, help='csv file to contain features')

    # preprocessing gene
    parser.add_argument("--gene_X", type=str, required=False, help="The unprocessed gene data.")
    parser.add_argument("--gene_Y", type=str, required=False, help="The unprocessed gene labels.")
    parser.add_argument('--gene_outputY', type=str, required=False, help='csv file to contain labels')
    parser.add_argument('--gene_outputX', type=str, required=False, help='csv file to contain features')

    # histone 
    parser.add_argument("--histone_features", type=str, required=False, help="The histone data to use for training.")
    parser.add_argument("--histone_labels", type=str, required=False, help="The histone labels to use for training")
    # gene
    parser.add_argument("--gene_features", type=str, required=False, help="The histone data to use for training.")
    parser.add_argument("--gene_labels", type=str, required=False, help="The histone labels to use for training")
    # both histone and gene
    parser.add_argument("--output_dir", type=str, required=False, help="The output directory for the models")

    # optional gene
    parser.add_argument("--hyperparam", action='store_true', help="Adding this argument will run the hyperparameter search on gene expr data")
    parser.add_argument("--load_new", action='store_false', help="Adding this argument will load the OLD format gene expr data")

    # optional to time or not
    parser.add_argument("--time", action='store_true', help="Adding this argument will time the runtimes")
    
    args = parser.parse_args()

    return args


def execute(file2run, arguments, suppress = False, time = False):
    
    if time:
        cmd = "time python3 "
    else:
        cmd = "python3 "
    if suppress:
        cmd += "-W ignore "
    cmd += file2run + arguments
    print(cmd)
    arch = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    print(arch.decode("utf-8"))


# TODO:
    # run gene, run histone, run pca
    # run preprocessing

def args_checker(args):

    # make sure have all the args to run the other python files
    if args.preprocess_gcp:
        if args.gcp_X is None or args.gcp_Y is None or args.gcp_outputY is None or args.gcp_outputX is None:
            raise Exception("Not enough arguments to process the GCP data")
        print("Will preprocess GCP Data")

    if args.preprocess_gene:
        if args.gene_X is None or args.gene_Y is None or args.gene_outputY is None or args.gene_outputX is None:
            raise Exception("Not enough arguments to process the Gene data")
        print("Will preprocess Gene Data")

    if args.run_histone:
        if args.histone_features is None or args.histone_labels is None or args.output_dir is None:
            raise Exception("Not enough arguments to analyze histone data")
        print("Will analyze Histone Data")

    if args.run_gene:
        if args.gene_features is None or args.gene_labels is None or args.output_dir is None:
            raise Exception("Not enough arguments to analyze gene data")
        print("Will analyze Gene Data")

    if args.run_pca:
        if args.histone_features is None or args.histone_labels is None or args.output_dir is None:
            raise Exception("Not enough arguments to analyze gene data")
        print("Will run PCA")

    


def main():

    args = get_args()

    args_checker(args)

    time = args.time
    if time:
        print("Timing enabled")

    print()

    if args.preprocess_gcp:

        file2run = "preprocess/GCP/extractData.py "
        arguments = "--X " + args.gcp_X + " --Y " + args.gcp_Y + " --outputY " + args.gcp_outputY + \
            " --outputX " + args.gcp_outputX

        print("Preprocessing GCP data")
        execute(file2run, arguments, suppress = False, time = time)

    if args.preprocess_gene:

        file2run = "preprocess/Gene_Expr/extractData_expr.py "
        arguments = "--X " + args.gene_X + " --Y " + args.gene_Y + " --outputY " + args.gene_outputY + \
            " --outputX " + args.gene_outputX

        print("Preprocessing Gene data")
        execute(file2run, arguments, suppress = False, time = time)


    if args.run_histone:

        file2run = "models/histone.py "
        arguments = "--features " + args.histone_features + " --labels " + args.histone_labels + " --output_dir " + args.output_dir

        print("Analyzing Histone data")
        execute(file2run, arguments, suppress = False, time = time)


    if args.run_gene:

        hyper = 0
        load = 0
        if args.hyperparam:
            hyper = 1
        if args.load_new:
            load = 1


        file2run = "models/gene.py "
        arguments = "--features " + args.gene_features + " --labels " + args.gene_labels + " --output_dir " + args.output_dir + " --hyperparam " + str(hyper) + " --load_new " + str(load)


        print("Analyzing Gene data")
        execute(file2run, arguments, suppress = True, time = time)
        
    if args.run_pca:

        file2run = "models/pca.py "
        arguments = "--features " + args.histone_features + " --labels " + args.histone_labels + " --output_dir " + args.output_dir

        print("Running PCA")
        execute(file2run, arguments, suppress = True, time = time)


if __name__ == "__main__":
    main()