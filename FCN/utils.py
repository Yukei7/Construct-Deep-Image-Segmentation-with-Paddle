
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.avg = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.cnt += n
        self.total += val * n
        self.avg = self.total / self.cnt

