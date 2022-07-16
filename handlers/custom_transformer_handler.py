
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
    This handler takes in a input string and multiple parameters and returns autoregressive generations from various OPT models. 
    """
    def __init__(self):
        super(TransformersHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """ 
        The function looks at the specs of the device that is running the server and loads in the model and any other objects that must be loaded in.
        
        """
        # get the passed properties of the torchserve compiler and the device 
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
        """
        The initial entry of data being passed for inference. 
        Here it is where we extract the parameters and inputs. 
        Inputs are tokenized for inference.
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
        """
        Function for performing inference on the processed input. The predictions are then decoded and returned.
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
        '''
        Extra function for processing inference outputs if not already done so.
        '''
        return inference_output
