import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import Constants
from dataset import TranslationDataset, paired_collate_fn


def cal_performance(pred, tgt, smoothing=False):


def cal_loss(pred, tgt, smoothing):
	tgt = tgt.contiguous().view(-1)

	if smoothing:
	
	else:
		
