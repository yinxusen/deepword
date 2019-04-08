import numpy as np

from deeptextworld.log import Logging


class FloorPlanCollector(Logging):
    def __init__(self):
        super(FloorPlanCollector, self).__init__()
        # collections of all actions and its indexed vectors
        self.fp_base = {}

        # current episode actions
        self.curr_fp = {}
        self.curr_eid = None

    def init(self):
        self.curr_fp = {}
        self.curr_eid = None

    def add_new_episode(self, eid):
        if eid == self.curr_eid:
            self.info("continue current episode: {}".format(eid))
            return

        self.info("add new episode in floor plan: {}".format(eid))

        if self.curr_eid is not None:
            if self.curr_eid not in self.fp_base:
                self.fp_base[self.curr_eid] = {}
            self.fp_base[self.curr_eid].update(self.curr_fp)

        self.init()
        self.curr_eid = eid
        if self.curr_eid in self.fp_base:
            self.info("found existing episode: {}".format(self.curr_eid))
            self.curr_fp = self.fp_base[self.curr_eid]
            self.info("{} floor paths loaded".format(len(self.curr_fp)))
        else:
            pass

    def extend(self, fps):
        for fp in fps:
            p1, d, p2 = fp
            if p1 not in self.curr_fp:
                self.curr_fp[p1] = {}
            if d not in self.curr_fp[p1]:
                self.info("find new path: {} + {} -> {}".format(p1, d, p2))
                self.curr_fp[p1][d] = p2
            else:
                if p2 != self.curr_fp[p1][d]:
                    self.error("mismatch floor plan: {} + {} -> {}, change to {}".format(
                        p1, d, self.curr_fp[p1][d], p2))
                else:
                    pass

    def get_map(self, room):
        if room is None or room not in self.curr_fp:
            return "floor plan : unknown"
        else:
            return "floor plan : {}".format(
                " , ".join(map(lambda d_r: '{} = {}'.format(d_r[0], d_r[1]),
                               list(self.curr_fp[room].items()))))

    def save_fps(self, path):
        if self.curr_eid is not None:
            self.fp_base[self.curr_eid] = self.curr_fp
        np.savez(path, fp_base=list(self.fp_base.items()))

    def load_fps(self, path):
        saved = np.load(path)
        fp_base = dict(saved["fp_base"])
        self.fp_base.update(fp_base)
