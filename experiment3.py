
import os
import json
import pathlib
import dataclasses
from datetime import datetime
from maestrodata import *
from mtransformer import *

def main(args):

    random_state = np.random.RandomState(args.seed)
    maestro_config = MaestroDatasetConfig()
    maestro_config.years = None
    maestro_config.sec_per_sample = [1, 3]
    maestro_config.max_source_length = 130
    maestro_config.max_target_length = 800
    maestro_data = MaestroData(maestro_config, random_state=random_state)

    train_data = MaestroDataset(maestro_data, batch_size=1, max_size=args.max_size, train=True, fixed_sample=args.fixed_sample)
    valid_data = MaestroDataset(maestro_data, batch_size=1, max_size=args.max_size, validation=True, fixed_sample=args.fixed_sample)

    print(f"#records in training set: {len(train_data.records)}")
    print(f"#records in validation set: {len(valid_data.records)}")

    configuration = {
            'args' : vars(args),
            'dataset' : dataclasses.asdict(maestro_config),
            'other' : {
                '#records in training set': len(train_data.records),
                '#records in validation set': len(valid_data.records)
            }
    }

    workers = min(os.cpu_count(), 40)
    train_generator = torch.utils.data.DataLoader(train_data, worker_init_fn=train_data.worker_init_fn,
            batch_size=args.batch_size, num_workers=workers)
    valid_generator = torch.utils.data.DataLoader(valid_data, worker_init_fn=valid_data.worker_init_fn,
            batch_size=args.batch_size, num_workers=workers)

    dtstamp = datetime.now().strftime('%Y.%m.%d-%H.%M')
    experiment = f'experiment3/{dtstamp}'
    checkpoint_dir = f'./checkpoint/{experiment}'
    checkpoint_name = '{epoch}_{training_loss_epoch:.1e}'
    logger_dir = f'./logs/{experiment}'
    other_dir = f'./experiments/{experiment}'
    config_log_file = f'{other_dir}/config.json'

    # os.makedirs(other_dir)
    # with open(config_log_file, "w") as text_file:
    #     text_file.write(json.dumps(configuration, indent=4))

    # os.makedirs(checkpoint_dir)
    # checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_weights_only=False,
    #     every_n_epochs=args.checkpoint_every_n_epoch, filename=checkpoint_name,
    #     save_top_k=args.save_top_k)
    # logger = TensorBoardLogger(logger_dir, name="default", flush_secs=5)

    model = Model.load_from_checkpoint(args.checkpoint, config=maestro_config)
    print(f'restored model from {args.checkpoint}')

    x = next(iter(valid_generator))
    model = model.to(args.device)
    mfcc = { 'mfcc' : x['mfcc'].to(args.device) }
    yp = model.generate(mfcc, 0, use_cache=args.use_cache, tqdm=tqdm if args.pbar else None)

    yy = pd.DataFrame({'prediction':yp.to('cpu')[0], 'label':x['label'][0]})
    print(yy)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--seed', type=int, default=2021)
    #parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_size', type=int, default=1)
    #parser.add_argument('--lr', type=float)
    #parser.add_argument('--checkpoint_every_n_epoch', type=int, default=10)
    #parser.add_argument('--save_top_k', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pbar', dest='pbar', action='store_true')
    parser.add_argument('--no-pbar', dest='pbar', action='store_false')
    parser.set_defaults(pbar=True)
    parser.add_argument('--fixed-sample', dest='fixed_sample', action='store_true')
    parser.add_argument('--no-fixed-sample', dest='fixed_sample', action='store_false')
    parser.set_defaults(fixed_sample=True)
    parser.add_argument('--validate', dest='validate', action='store_true')
    parser.add_argument('--no-validate', dest='validate', action='store_false')
    parser.set_defaults(validate=True)
    parser.add_argument('--use-cache', dest='use_cache', action='store_true')
    parser.add_argument('--no-use-cache', dest='use_cache', action='store_false')
    parser.set_defaults(use_cache=True)

    args = parser.parse_args()
    if not args.checkpoint is None and not ( os.path.exists(args.checkpoint) and os.path.isfile(args.checkpoint) ):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.checkpoint)
    main(args)

