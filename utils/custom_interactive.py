import torch
import logging
import os
import sys


from fairseq import checkpoint_utils, options, tasks, utils
from collections import namedtuple

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


class Generator():
    def __init__(self, data_path, checkpoint_path="checkpoint_best.pt"):
        #self.parser = options.options.get_interactive_generation_parser()
        self.parser = options.get_generation_parser(interactive=True)
        self.parser.set_defaults(path=checkpoint_path,
            remove_bpe="gpt2", num_wokers=5
        )
        self.args = options.parse_args_and_arch(self.parser, 
            input_args=[data_path]
        )
        

        print("parser: ", self.parser)
        #self.args.data = '/content/drive/MyDrive/Modified-models/Implement_PALM/corpus/cnn_dm_bin'
        
        self.args.user_dir = "/content/drive/MyDrive/Modified-models/Implement_PALM/PALM/src"
        self.args.truncate_source = True
        self.args.source_lang = "source"
        self.args.target_lang = "source"
        self.args.task = "auto_encoding_regressive"
        self.args.remove_bpe = "gpt2"
        self.args.beam = 4
        self.args.nbest = 1
        self.args.lenpen = 0.6
        
        utils.import_user_module(self.args)
        
        if self.args.buffer_size < 1:
            self.args.buffer_size = 1
        if self.args.max_tokens is None:
            self.args.max_sentences = 1

        assert not self.args.sampling or self.args.nbest == self.args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        #assert not self.args.max_sentences or self.args.max_sentences <= self.args.buffer_size, \
        #    '--max-sentences/--batch-size cannot be larger than --buffer-size'

        self.use_cuda = torch.cuda.is_available() and not self.args.cpu

        self.task = tasks.setup_task(self.args)

        self.models, self._model_args = checkpoint_utils.load_model_ensemble(
            self.args.path.split(':'),
            arg_overrides=eval(self.args.model_overrides),
            task=self.task,
        )

        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                need_attn=self.args.print_alignment,
            )
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        print("self.args ", self.args)
        print("find gen: ", self.generation)
        self.generator = self.task.build_generator(self.args)

        if self.args.remove_bpe == 'gpt2':
            from fairseq.gpt2_bpe.gpt2_encoding import get_encoder
            self.decoder = get_encoder(
                'fairseq/gpt2_bpe/encoder.json',
                'fairseq/gpt2_bpe/vocab.bpe',
            )
            self.encode_fn = lambda x: ' '.join(map(str, self.decoder.encode(x)))
        else:
            self.decoder = None
            self.encode_fn = lambda x: x

        self.align_dict = utils.load_align_dict(self.args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models]
        )
        print("done")


if __name__ == '__main__':
    gen = Generator('/content/drive/MyDrive/Modified-models/Implement_PALM/corpus/cnn_dm_bin', 
                    "/content/drive/MyDrive/test_tensorboard/palm/palm_cnn_dm_checkpoints/checkpoint_best.pt")

