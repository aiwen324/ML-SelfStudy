import torch
import torch.nn as nn
import Constants
import argparse




def load_insts_from_files(file, max_sent_len):
	""" Read the sentence fro spcified path 
		return a list of splitted sentence
	"""
	word_insts = []
	trimeed_inst_count = 0
	with open(file) as f:
		for inst in f:
			inst = inst.split()
			if len(inst) > max_sent_len:
				trimeed_inst_count += 1
				inst = inst[:max_sent_len]
			word_insts += [[Constants.BOS] + inst + [Constants.EOS]]
			return word_insts



def filtered_vocabs(word_insts, min_requirement=1):
	""" Generate a word dictionary
		Would get rid off some words based on the
		min_requirement of number of appearance
	"""
	vocabs = set(w for inst in word_insts for w in inst)
	word_count_dict = {w : 0 for w in vocabs}
	for inst in word_insts:
		for w in inst:
			word_count_dict[w] += 1
	filtered_vocabs = set()
	for w in word_count_dict:
		if word_count_dict[w] >= min_requirement
			filtered_vocabs.add(w)
	filtered_vocabs.remove(Constants.BOS_, Constants.EOS_)

	print("Totally get {} vocabs.".format(len(vocabs)))
	print("Filtered vocab size for minimum appearance requirement is \
		{}".format(len(filtered_vocabs)))

	return filtered_vocabs

def generate_word_idx(vocabs):
	word_to_idx = {Constants.BOS_ : Constants.BOS, 
	               Constants.EOS_ : Constants.EOS,
	               Constants.UNK_ : Constants.UNK,
	               Constants.PAD_ : Constants.UNK}
	# TODO: Fix the iteration syntax
	for w in vocabs:
		if w not in word_to_idx:
			word_to_idx[w] = len(word_to_idx)
	return word_to_idx

def transform_word_to_idx(insts, word_to_idx):
	idx_insts = []
	for inst in insts:
		l = []
		for w in insts:
			try:
				l.append(word_to_idx[w])
			except KeyError:
				l.append(Constants.UNK_)
		idx_insts.append(l)
	return idx_insts
 	

def main():
	parser = argparse.ArgumentParser(description=
									 "Preprocessing Language Corpus")
    # parser.add_argument('index', default=0, type=int,
    #                     help="Image index to plot")
    # parser.add_argument('--checkpoint', default="",
    #                     help="Model file to load and save")
    # parser.add_argument('--outdir', default="outputs/act",
    #                     help="Directory to save the file")
    parser.add_argument('source_data', help="Path of source data")
    parser.add_argument('target_data', help="Path of target data")
    parser.add_argument('max_sent_len', default=50, type=int,
    					help="Maximum Length of sentence sequence")
    parser.add_argument('save_path', help="Path for save the loaded data."+
    									  "It will be in Torch Format")
    
    opt = parser.parse_args()
    src_word_insts = load_insts_from_files(opt.source_data, opt.max_sent_len)
    src_filtered_vocabs = filtered_vocabs(src_word_insts)
    src_word_to_idx =generate_word_idx(src_filtered_vocabs)
    src_idx_insts = transform_word_to_idx(src_word_insts, src_word_to_idx)

    tgt_word_insts = load_insts_from_files(opt.target_data, opt.max_sent_len)
    tgt_filtered_vocabs = filtered_vocabs(tgt_word_insts)
    tgt_word_to_idx =generate_word_idx(tgt_filtered_vocabs)
    tgt_idx_insts = transform_word_to_idx(tgt_word_insts, tgt_word_to_idx)

    # TODO: Add shuffle mechanism to take the validation data
    total_sample_size = len(src_word_insts)
    assert total_sample_size == len(tgt_word_insts)
	indices = np.arange(total_sample_size)
	np.random.shuffle(indices)
	train_size = int(total_sample_size * 0.8)
	
    # TODO: Save the data
	data = {
		'settings': opt,
		'dict': {
		   	'src' : src_word_to_idx,
		   	'tgt' : tgt_word_to_idx},
		'train': {
			'src' : src_word_insts[indices[:train_size]],
			'tgt' : tgt_word_insts[indices[:train_size]]},
		'valid': {
			'src' : src_word_insts[indices[train_size:]],
			'tgt' : tgt_word_insts[indices[train_size:]]}
			}
	print('[Info] Dumping the processed data to file', opt.save_path)
	torch.save(data, opt.save_path)
	print('[Info] Finish')

if __name__ == '__main__':
	main()