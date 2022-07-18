# Vertex 4 Dummies

<p align="center">
  <img src="https://raw.githubusercontent.com/jamesonhohbein/vertex_deployment_4dummies/main/logo4.png" />
</p>

## Requirements 
- Docker: You need to be working in an environment where docker is accessible. 
- GCP SDK Authentication: You need to be authenticated to make changes to a GCP project via the GCP Python SDK. (I recommend running in a Vertex AI workbench notebook!)
## Usage 
Install Package and import functions 

```
!pip install git+https://github.com/jamesonhohbein/vertex_deployment_4dummies

from vertex4dummies.build_script import build_huggingface_autoregressive_deployment,call_inference
```

Deploy Model 
```
build_huggingface_autoregressive_deployment(model_link='https://huggingface.co/facebook/opt-350m',
                                            PROJECT_ID='myprojectID',
                                            APP_NAME='opt350m',
                                            VERSION=1.0,
                                            HANDLER='custom_transformer_handler.py',
                                            machine_type='n1-standard-16',
                                            DESCRIPTION='Opt-350m')
                                            
```

Call inference

```
t = [
    {
        "data": {
            "text": "This is a test of"
        },
        "parameters":{
            "num_return_sequences": 5,
            "top_p":0.9,
            "top_k":10,
            "temperature":0.8,
            "max_length":20,
            "no_repeat_ngram_size":2
            
            
        }
    }
]
call_inference(myendpointresource,t)
```
