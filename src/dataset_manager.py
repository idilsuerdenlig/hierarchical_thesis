class DatasetManager():
    def __init__(self):
        self.s = None
        self.a = None
        self.r = None
        self.ss = None
        self.abs = None
        self.last = None
        self.dataset = list()

    def add_first_sample(self, sample, skip):
        if not skip:
            self.s = None
            self.a = sample[1]
            self.r = None
            self.ss = sample[0]
            self.abs = None
            self.last = None

    def add_sample(self, sample, skip):
        if not skip:
            self.s = self.ss
            self.ss, a, self.r, self.abs, self.last = sample
            sample_step = self.s, self.a, self.r, self.ss, self.abs, self.last
            self.dataset.append(sample_step)
            self.a = a

    def empty_dataset(self):
        self.dataset = list()

    def get(self):
        return self.dataset
