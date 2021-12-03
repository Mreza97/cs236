
import os
import glob
import json
import pathlib
import dataclasses
from datetime import datetime
from maestrodata import *
from mtransformer import *

def main(args):

    random_state = None #np.random.RandomState(args.seed)
    maestro_config = MaestroDatasetConfig()
    maestro_config.years = None
    maestro_config.sec_per_sample = [1, 3]
    maestro_config.max_source_length = 130
    maestro_config.max_target_length = 800
    maestro_data = MaestroData(maestro_config, random_state=random_state)

    valid_data = MaestroDataset(maestro_data, batch_size=1, max_size=None, validation=True, fixed_sample=False)

    print(f"#records in validation set: {len(valid_data.records)}")

    configuration = {
            'args' : vars(args),
            'dataset' : dataclasses.asdict(maestro_config),
            'other' : {
                '#records in validation set': len(valid_data.records)
            }
    }

    workers = min(os.cpu_count(), 20)
    valid_generator = torch.utils.data.DataLoader(valid_data, worker_init_fn=valid_data.worker_init_fn,
            batch_size=args.batch_size, num_workers=workers)

    dtstamp = datetime.now().strftime('%Y.%m.%d-%H.%M')
    experiment = f'experiment3/{dtstamp}'
    #checkpoint_dir = f'./checkpoint/{experiment}'
    #checkpoint_name = '{epoch}_{training_loss_epoch:.1e}'
    logger_dir = f'./logs/{experiment}'
    other_dir = f'./experiments/{experiment}'
    config_log_file = f'{other_dir}/config.json'

    if not os.path.exists(other_dir):
        os.makedirs(other_dir)
    with open(config_log_file, "w") as text_file:
        text_file.write(json.dumps(configuration, indent=4))

    #os.makedirs(checkpoint_dir)
    #checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_weights_only=False,
    #    every_n_epochs=args.checkpoint_every_n_epoch, filename=checkpoint_name,
    #    save_top_k=args.save_top_k)
    #logger = TensorBoardLogger(logger_dir, name="default", flush_secs=5)

    names = {
        '2021.11.27-00.16': 0,
        '2021.11.27-15.11': 200,
        '2021.11.28-08.27': 400,
        '2021.11.28-21.40': 600,
        '2021.11.29-09.40': 800,
        '2021.11.29-21.23': 1000,
        '2021.11.30-09.37': 1200,
        '2021.11.30-21.35': 1400 }

    ckpt_pattern = f"./checkpoint/experiment2/*/epoch=*.ckpt"
    files = [ (_.split('=')[1].split('_')[0],_) for _ in glob.glob(ckpt_pattern) ]
    files.sort(key=lambda x:int(x[0]))

    checkpoints = []
    for e,checkpoint in files:
        dir = os.path.basename(os.path.dirname(checkpoint))
        epoch = int(e) + names[dir]
        checkpoints.append((epoch, checkpoint))

    checkpoints.sort(key=lambda x:int(x[0]))
    for epoch, checkpoint in checkpoints:
        model = Model.load_from_checkpoint(checkpoint, config=maestro_config)
        logger = TensorBoardLogger(logger_dir, name=f"epoch-{epoch:04d}", flush_secs=5)
        trainer = Trainer(log_every_n_steps=min(len(train_generator), 50),
                          devices=1, accelerator="gpu",
                          default_root_dir="./", logger=logger,
                          enable_progress_bar=args.pbar)
        #trainer.validate(model, val_dataloaders=valid_generator)
        print(epoch, checkpoint)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--checkpoint_every_n_epoch', type=int, default=10)
    parser.add_argument('--save_top_k', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--pbar', dest='pbar', action='store_true')
    parser.add_argument('--no-pbar', dest='pbar', action='store_false')
    parser.set_defaults(pbar=True)
    parser.add_argument('--validate', dest='validate', action='store_true')
    parser.add_argument('--no-validate', dest='validate', action='store_false')
    parser.set_defaults(validate=True)

    args = parser.parse_args()
    if not args.checkpoint is None and not ( os.path.exists(args.checkpoint) and os.path.isfile(args.checkpoint) ):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.checkpoint)
    main(args)

