import yaml
import wandb
from utils import load_config
from architectures.alpha4 import self_play, pretrain_model
from architectures.alpha5 import DQNAgent
from architectures.alpha6 import ChessEnv, ChessAgent

if __name__ == '__main__':

    config = load_config("config.yaml")
    model = config['model']
    
    wandb.login() 
    wandb.init(project = 'chess-qnn',
        name = model['run_name'],
        config = config)
    
    if config['alpha5']['enabled']: 
        alpha5 = config['alpha5']
        agent = DQNAgent()
        if model['load']:
            agent.load_pretrained_weights("/models/" + model['name'])
        
        if alpha5['pretraining']['doit']:
            pretraining = alpha5['pretraining']
            agent.supervised_train(dataset_path = pretraining['data_path'], 
                                epochs = pretraining['epochs'], 
                                batch_size = pretraining['batch_size'],
                                save_path = "/models/" + model['name'] + "_pretrained.h5")
        
        if alpha5['training']['doit']:
            training = alpha5['training']
            agent.train(episodes = training['episodes'], 
                        batch_size = training['batch_size'],
                        save_path = "/models/" + model['name'] + ".h5")


