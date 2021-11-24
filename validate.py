
import os
import errno
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
    maestro_config.max_source_length = 129
    maestro_config.max_target_length = 800
    maestro_data = MaestroData(maestro_config, random_state=random_state)

    #train_data = MaestroDataset(maestro_data, batch_size=1, max_size=args.max_size, train=True, fixed_sample=False)
    valid_data = MaestroDataset(maestro_data, batch_size=1, max_size=args.max_size, validation=True, fixed_sample=False)

    #print(f"#records in training set: {len(train_data.records)}")
    print(f"#records in validation set: {len(valid_data.records)}")

    workers = min(os.cpu_count(), 40)
    #train_generator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=workers)
    valid_generator = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=workers)

    dtstamp = datetime.now().strftime('%Y.%m.%d-%H.%M')

    model = Model.load_from_checkpoint(args.checkpoint, config=maestro_config)
    print(f'restored model from {args.checkpoint}')

    trainer = Trainer(devices=1, accelerator="gpu", default_root_dir="./", enable_progress_bar=args.pbar)

    trainer.validate(model, valid_generator)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--max_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
    parser.add_argument('--pbar', dest='pbar', action='store_true')
    parser.add_argument('--no-pbar', dest='pbar', action='store_false')
    parser.set_defaults(pbar=True)

    args = parser.parse_args()
    if not args.checkpoint is None and not ( args.checkpoint.exists() and args.checkpoint.is_file() ):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.checkpoint)

    main(args)

