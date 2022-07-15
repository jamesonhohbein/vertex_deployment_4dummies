
import os
import json
import logging
from os import listdir

import torch
from transformers import AutoTokenizer, OPTForCausalLM
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersHandler(BaseHandler):
    """
    The handler takes an input string and returns the classification text 
    based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """ Loads the model.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt or pytorchf_model.bin file")
        
        # Load model
        logger.info("Loading Model...")
        self.model = OPTForCausalLM.from_pretrained(model_dir)
        logger.info("Model loaded...")
        
        self.model.to(self.device)
        
        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))
        
        # Ensure to use the same tokenizer used during training
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info("Tokenizer loaded")

        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """
        params = data[0].get("parameters")
        text = data[0].get("data").get('text')
        
        # set the params 
        self.num_return_sequences = params.get('num_return_sequences')
        self.top_p = params.get('top_p')
        self.top_k = params.get('top_k')
        self.temperature = params.get('temperature')
        self.max_length = params.get('max_length')
        self.no_repeat_ngram_size = params.get('no_repeat_ngram_size')
        
        inputs = self.tokenizer(text, return_tensors='pt')

        return inputs

    def inference(self, inputs):
        """ Predict the class of a text using a trained transformer model.
        """
        
        prediction = self.model.generate(inputs.input_ids,
                                         max_length=self.max_length,
                                         num_return_sequences= self.num_return_sequences,
                                         do_sample = True,
                                         temperature = self.temperature,
                                         early_stopping = True,
                                         top_k = self.top_k,
                                         top_p = self.top_p,
                                         no_repeat_ngram_size = 2,
                                         return_dict_in_generate = True,
                                         tokenizer = self.tokenizer)
        
        
        prediction = self.tokenizer.batch_decode(prediction['sequences'],
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
        
        return [prediction]

    def postprocess(self, inference_output):
        return inference_output
