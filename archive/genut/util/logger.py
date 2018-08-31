import time
import torch
import sys


def show_progress_bar(epo, current_batch, total_batch, time_start, time_current, data_string):
    sys.stdout.write('\r')
    hours, rem = divmod(time_current - time_start, 3600)
    minutes, seconds = divmod(rem, 60)
    rate = int(40 * current_batch / total_batch)
    progress = int(float(current_batch) / float(total_batch) * 100)
    sys.stdout.write(
        "Epo: %2d [%-40s] %3d%% %-2d:%-2d:%02.0f %s" % (
            epo, '=' * rate, progress, int(hours), int(minutes), seconds, data_string))
    sys.stdout.flush()


class LossItem():
    def __init__(self, name, node, weight=1):
        self.name = name
        self.weight = weight
        self.node = node
        if type(node) == torch.autograd.Variable:
            self.val = node.data[0]
        else:
            self.val = node


class LossManager():
    """
    Given we have many kind of loss(es), we need a class to manage these loss to combine, visualize, weight.
    LossManager is initialized at the beginning of every epo, storing all loss value for this epo and ONLY computation
    graph for this batch.
    """

    def __init__(self):
        """"""
        self.history_loss = {}

        self.cur_batch_lossItem = []

    def compute(self):
        _loss = Var(torch.zeros(1)).cuda()
        for item in self.cur_batch_lossItem:
            if item.weight < 0.01:
                continue
            _loss += item.node * item.weight
        self.add_LossItem(LossItem(name='ALL', node=_loss, weight=1))
        return _loss

    def visual(self, epo_idx, cnt, n_batch, start_time, time_end):
        str_to_write = ''
        for k, v in self.history_loss.iteritems():
            val = sum(v) / len(v)
            str_to_write += ' %s:%02.03f ' % (k, val)

        show_progress_bar(epo_idx, cnt, n_batch, start_time, time_end,
                          str_to_write)

    def add_LossItem(self, li):
        self.cur_batch_lossItem.append(li)

    def clear_cache(self):
        for idx, item in enumerate(self.cur_batch_lossItem):
            name = item.name
            if self.history_loss.has_key(name):
                self.history_loss[name].append(item.val)
            else:
                self.history_loss[name] = [item.val]
            if len(self.history_loss[name]) > 50:
                self.history_loss[name] = self.history_loss[name][-50:]
        self.cur_batch_lossItem = []


class Logger(object):
    def __init__(self, print_every, num_all_batch):
        self.print_every = print_every
        self.n_batch = num_all_batch

    def init_new_epo(self, epo):
        self.current_epo = {}
        self.current_epo['count'] = 0
        self.current_epo['epo'] = epo
        # self.current_epo['max_len_enc'], self.current_epo['max_len_dec'], self.current_epo[
        #     'cov_loss_weight'] = schedule(epo)
        self.current_epo['time_start'] = time.time()
        self.lm = LossManager()

    def init_new_batch(self, batch_id):
        self.current_batch = {'id': batch_id}
        self.current_epo['count'] += 1

    def set_oov(self, max_oov_len):
        self.current_batch['max_oov_len'] = max_oov_len

    def batch_end(self):
        # call LossManager compute
        # Different loss has been added to list in LM before
        # Compute is to get the final loss for optimization
        self.lm.clear_cache()
        if self.current_epo['count'] % self.print_every == 0:
            cnt = self.current_epo['count']
            # call LossManager visual
            time_end = time.time()
            self.lm.visual(self.current_epo['epo'], cnt + 1, self.n_batch, self.current_epo['time_start'],
                           time_end)
