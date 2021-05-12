import time
import sys
import logging
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv


def export_plot(ys, ylabel, filename):
    """
    Export a plot in filename

    Args:
        ys: (list) of float / int to plot
        filename: (string) directory
    """
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


class CSVLogger():
    """Credit to Jonathan Lorraine"""

    def __init__(self, fieldnames, filename='log.csv'):
        self.filename = filename
        self.csv_file = open(filename, 'w')
        self.writer = csv.DictWriter(self.csv_file,
                                     fieldnames=fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()

    def writerow(self, row):
        """
        :param row:
        :return:
        """
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        """
        :return:
        """
        self.csv_file.close()


def get_logger(filename):
    """
    Return a logger instance to a file
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def load_logger(filename):
    """Setup a csv logger for saving data.

    :param args: The arguments to the experiment.
    :param fieldnames: The names of data to be stored.
    :return: The csv_logger object.
    """
    data_dict = {
        'Timestep': 0,
        'Training Rewards': 0,
        'Max Q': 0,
        'Eval Rewards': 0,
        'Loss': 0
    }
    fieldnames = [key for key, _ in data_dict.items()]
    csv_logger = CSVLogger(fieldnames=fieldnames, filename=filename)
    return csv_logger


def load_from_csv(filename):
    """
    :param path:
    :param do_train:
    :param do_val:
    :param do_test:
    :return:
    """
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                data[name].append(row[name])
    data['Timestep'] = np.array([int(i) for i in data['Timestep']])
    data['Training Rewards'] = np.array([float(i) for i in data[
        'Training Rewards']])
    data['Max Q'] = np.array([float(i) for i in data['Max Q']])
    data['Eval Rewards'] = np.array([float(i) for i in data[
        'Eval Rewards']])
    data['Loss'] = np.array([float(i) for i in data['Loss']])
    return data


def csv_plot(file_name, save_path):
    data = load_from_csv(file_name)

    plt.figure()
    plt.plot(data['Timestep'], data['Training Rewards'])
    plt.xlabel("Timesteps")
    plt.ylabel('Training Rewards')
    plt.grid()
    plt.savefig(save_path + 'finaltrainrewards.pdf')
    plt.close()

    plt.figure()
    plt.plot(data['Timestep'], data['Eval Rewards'])
    plt.xlabel("Timesteps")
    plt.ylabel('Eval Rewards')
    plt.grid()
    plt.savefig(save_path + 'evalrewards.pdf')
    plt.close()

    plt.figure()
    plt.plot(data['Timestep'], data['Max Q'])
    plt.xlabel("Timesteps")
    plt.ylabel('Max Q')
    plt.grid()
    plt.savefig(save_path + 'maxQ.pdf')
    plt.close()

    plt.figure()
    plt.plot(data['Timestep'], data['Loss'])
    plt.xlabel("Timesteps")
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(save_path + 'loss.pdf')
    plt.close()


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, discount=0.9):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.exp_avg = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self.discount = discount

    def reset_start(self):
        self.start = time.time()

    def update(self, current, values=[], exact=[], strict=[],
               exp_avg=[], base=0):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [
                    v * (current - self.seen_so_far),
                    current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (
                            current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v
        for k, v in exp_avg:
            if k not in self.exp_avg:
                self.exp_avg[k] = v
            else:
                self.exp_avg[k] *= self.discount
                self.exp_avg[k] += (1 - self.discount) * v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / (current - base)
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][
                        0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            for k, v in self.exp_avg.items():
                info += ' - %s: %.4f' % (k, v)

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][
                        0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)
