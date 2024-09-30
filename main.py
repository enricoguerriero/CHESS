import yaml
import wandb
from utils import load_config
from architectures.nn_torch_1 import DQNAgent


if __name__ == '__main__':

    model, data, pretraining, training = load_config('config.yaml')
    print(model)
    
    wandb.login() 
    wandb.init(project = 'chess-qnn',
        name = model['run_name'])
    wandb.log({
        "data": data if pretraining['enabled'] else None,
        "pretraining": pretraining if pretraining['enabled'] else None,
        "training": training if training['enabled'] else None
    })

    agent = DQNAgent()
    if model['load']:
        agent.load_pretrained_weights("/models/" + model['name'])
    
    if pretraining['enabled']:
        agent.supervised_train(dataset_path = data['data_path'], 
                               epochs = pretraining['epochs'], 
                               batch_size = pretraining['batch_size'],
                               save_path = "/models/" + model['name'] + "_pretrained.h5")
    
    if training['enabled']:
        agent.train(episodes = training['episodes'], 
                    batch_size = training['batch_size'],
                    save_path = "/models/" + model['name'] + ".h5")


