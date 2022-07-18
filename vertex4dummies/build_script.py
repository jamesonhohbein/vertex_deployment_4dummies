


import os 
from google.cloud import aiplatform

def build_huggingface_autoregressive_deployment(model_link:str,PROJECT_ID:str,APP_NAME:str,VERSION:float,HANDLER:str,machine_type:str,DESCRIPTION:str):
    '''
    Central function to build and and deploy an autoregressive huggingface model into vertex ai 

    Future:
        Add options for accelerators by filling the accelerator type and count fields https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus

    params
        model_link:str this is the url to a huggingface model 
        PROJECT_ID:str the project ID of the GCP 
        GCS_MODEL_ARTIFACTS_URI:str the path in cloud storage where the model is stored 
        APP_NAME:str whatever you want to name the application
        VERSION:float whatever version you are uploading of this model 
        HANDLER:str path to the torchserve handler you wish to use with thise model 
        machine_type:str the machine type to host the endpoint, found here https://cloud.google.com/vertex-ai/docs/predictions/configure-compute
        DESCRIPTION:str the description of the model 
    '''


    # download the model from huggingface 
    os.system("git lfs install")
    os.system(f"git clone {model_link}")

    # create the docker folder 
    os.system("mkdir predictor")

    # copy the handler 
    os.system(f"cp {HANDLER} predictor/")

    # set the custom predict image URI
    CUSTOM_PREDICTOR_IMAGE_URI = f"gcr.io/{PROJECT_ID}/pytorch_predict_{APP_NAME}"

    # get the name of the model 
    model_folder = model_link.split('/')[model_link.count('/')]


    # rename the folder 
    os.system(f"mv {model_folder} predictor/model")

    # run torchserve 
    model_source = ''
    for i in os.listdir('predictor/model'):
        if i[-4:] == '.bin':
            model_source = 'predictor/model/'+i

    # compile the model in torch serve 
    #os.system(f"torch-model-archiver --model-name {APP_NAME} --version {VERSION} --model-file {model_source} --serialized-file {model_source} --handler {HANDLER}")

    handler_name = HANDLER.split('/')[HANDLER.count('/')]

    extra_files = []

    # remove other saved models 
    [os.remove('predictor/model/'+x) for x in os.listdir('predictor/model') if '.h5' in x or '.msgpack' in x]

    # append names of extra file locations for when they are transfered. Do not include other model savings such as tensorflow or flax models. This is for efficiency. 
    [extra_files.append('/home/model-server/'+x) for x in os.listdir('predictor/model')]


    # format the docker file 
    docker_file = '''   
bash -s $APP_NAME

APP_NAME=$1

cat << EOF > ./predictor/Dockerfile

FROM pytorch/torchserve:latest-cpu

# install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install transformers
RUN pip3 install torch

USER model-server

# copy model artifacts, custom handler and other dependencies
COPY {0} /home/model-server/
COPY ./model/ /home/model-server/

# create torchserve configuration file
USER root
RUN printf "\\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f   --model-name={1}   --version={7}   --serialized-file=/home/model-server/{4}   --handler=/home/model-server/{5}   --extra-files "{6}"   --export-path=/home/model-server/model-store

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve",      "--start",      "--ts-config=/home/model-server/config.properties",      "--models",      "{2}={3}.mar",      "--model-store",      "/home/model-server/model-store"]

EOF

echo "Writing ./predictor/Dockerfile"
    '''.format(handler_name,APP_NAME,APP_NAME,APP_NAME,model_source.split('/')[model_source.count('/')],handler_name,','.join(extra_files),str(VERSION))

    #write docker file
    os.system(docker_file)

    
    # build docker image 
    os.system(f"docker build --tag={CUSTOM_PREDICTOR_IMAGE_URI} ./predictor")


    # push the docker image 
    os.system(f"docker push {CUSTOM_PREDICTOR_IMAGE_URI}")


    # init config model upload to vertex AI 
    aiplatform.init(project=PROJECT_ID)
    model_display_name = f"{APP_NAME}-v{VERSION}"
    model_description = DESCRIPTION

    MODEL_NAME = APP_NAME
    health_route = "/ping"
    predict_route = f"/predictions/{MODEL_NAME}"
    serving_container_ports = [7080]


    # upload model to vertex AI 
    print('\n\nDeploying Model to Vertex AI...\n')
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        description=model_description,
        serving_container_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
        serving_container_predict_route=predict_route,
        serving_container_health_route=health_route,
        serving_container_ports=serving_container_ports)

    model.wait()

    print('Model Display Name:',model.display_name)
    print('Resource name:',model.resource_name)

    # config and deploy endpoint to a model in vertex AI 
    endpoint_display_name = f"{APP_NAME}-endpoint"
    print('\n\nSpinning up endpoint\n')
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

    traffic_percentage = 100
    deployed_model_display_name = model_display_name
    sync = True
    print('\n\nConnecting Endpoint and Model, this may take a while...\n')
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        machine_type=machine_type,
        traffic_percentage=traffic_percentage,
        sync=sync)


callable_endpoint = None 

def call_inference(endpoint_resource_name:str,input:list):
    '''
    
    
    '''
    global callable_endpoint

    try: 
        if callable_endpoint.name not in endpoint_resource_name:
            callable_endpoint = aiplatform.Endpoint(endpoint_resource_name)
    except:
        callable_endpoint = aiplatform.Endpoint(endpoint_resource_name)


    prediction = callable_endpoint.predict(instances=input)

    return prediction 