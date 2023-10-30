import numpy as np
import torch
import random

class CurriculumSampler():
    # this supports progressive sampling of scalar values
    # the curriculum is Linear
    def __init__(self,
                 start_value,
                 end_value,
                 curriculum_length,
                 step_size=1,
                 randomly_backwards=False
                 ):

        self.start_value = start_value
        self.end_value = end_value
        self.curriculum_length = curriculum_length
        self.randomly_backwards = randomly_backwards
        self.step_size = step_size
        self._step = 0
        assert self.end_value > self.start_value
        assert curriculum_length > 0
        assert step_size > 0 and (self.end_value - self.start_value) % step_size == 0
        print('Length of curriculum: ', self.curriculum_length, 'Start value: ', self.start_value, 'End value: ', self.end_value, '')

    def get_value(self):
        current_value = self.start_value + \
                        int((self.end_value - self.start_value) // self.step_size * self._step / self.curriculum_length) *\
                        self.step_size
        if self.randomly_backwards:
            if np.random.uniform() < 0.3:
                current_value = np.random.randint(self.start_value, current_value) if current_value > self.start_value else current_value
        return current_value

    def step(self):
        self._step += 1
        if self._step > self.curriculum_length:
            self._step = self.curriculum_length

    def reset(self):
        self._step = 0