from abc import ABCMeta, abstractmethod
import random
import torch


class DataCenter(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.train = None
        self.dev = None
        self.test = None
