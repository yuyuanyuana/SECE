import os
import datetime 


class MakeLogClass:
    def __init__(self, log_file):
        self.log_file = log_file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
    def make(self, *args):
        print(*args)
        # Write the message to the file
        with open(self.log_file, "a") as f:
            for arg in args:
                f.write("{}\r\n".format(arg))


def construct_floder(path_name):
    result_path = path_name+'_'+datetime.datetime.now().strftime('%m-%d-%y %H:%M')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs(f'{result_path}/plot')
        os.makedirs(f'{result_path}/out')
        os.makedirs(f'{result_path}/loss')
        os.makedirs(f'{result_path}/annotation')
        os.makedirs(f'{result_path}/marker_plot')
    return result_path

