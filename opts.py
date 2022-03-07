import argparse

# our model related options
def parse_opts():
    parser = argparse.ArgumentParser(description = 'Our model options')

    parser.add_argument('--cuda', action = 'store_true', help = 'Choose device to use cpu cuda')

    parser.add_argument('--train_root', action = 'store', type = str,
                        default = '/home/yusha/deepthinkers/data/ILSVRC2012/data2017_downsampling/Train_781250', help = 'Root path of training data')

    parser.add_argument('--train_label_path', action = 'store', type = str, 
                        default = '/home/yusha/deepthinkers/data/ILSVRC2012/data2017_downsampling/Label/train_downsampling_781250.csv', help = 'Path of training data label')

    parser.add_argument('--test_root', action = 'store', type = str,
                        default = '/home/yusha/deepthinkers/data/ILSVRC2012/data2017_downsampling/Test_781250', help = 'Root path of test data')

    parser.add_argument('--test_label_path', action = 'store', type = str, 
                        default = '/home/yusha/deepthinkers/data/ILSVRC2012/data2017_downsampling/Label/test_downsampling_781250.csv', help = 'Path of test data label') 

    parser.add_argument('--mess_length', action = 'store', type = int, 
                        default = 233472, help = 'Data length, if data > mess_length, then truncation; else padding(233472)')

    parser.add_argument('--batch_size', action = 'store', type = int, default = 4, help = 'Batch size')

    parser.add_argument('--learning_rate', action = 'store', type  =float, default = 0.0001, help = 'initial learning rate')

    parser.add_argument('--epochs', action = 'store', type = int, default = 100, help = 'train rounds over training set')

    parser.add_argument('--classes_cavitation_detection', action = 'store', type = int, default = 4, help = 'total number of classes in dataset')

    parser.add_argument('--num_classes', action = 'store', type = int, default = 4, help = 'total number of classes in dataset')

    parser.add_argument('--n_threads', action = 'store', type = int, default = 1, help = 'Number of threads for multi-thread loading')

    parser.add_argument('--train_batch_shuffle', action = 'store', type = bool, default = True, help = 'Shuffle input batch for training data')

    parser.add_argument('--test_batch_shuffle', action = 'store', type = bool, default = False, help = 'Shuffle input batch for test data')

    args = parser.parse_args()
    return args