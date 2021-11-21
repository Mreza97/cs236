
from datetime import datetime
from maestrodata import *
from mtransformer import *

def main(args):

    random_state = np.random.RandomState(args.seed)
    maestro_config = MaestroDatasetConfig()
    maestro_config.years = [2017]
    maestro_config.sec_per_sample = [.5, .5]
    maestro_data = MaestroData(maestro_config, random_state=random_state)

    train_data = MaestroDataset(maestro_data, batch_size=1, max_size=1, train=True, fixed_sample=True)
    valid_data = MaestroDataset(maestro_data, batch_size=1, max_size=1, validation=True, fixed_sample=True)

    workers = min(os.cpu_count(), 40)
    train_generator = torch.utils.data.DataLoader(train_data, batch_size=8, num_workers=workers)
    valid_generator = torch.utils.data.DataLoader(valid_data, batch_size=8, num_workers=workers)

    dtstamp = datetime.now().strftime('%Y.%m.%d.%H.%M')
    experiment = f'overfit_{dtstamp}'
    checkpoint_dir = f'./checkpoint/{experiment}'
    checkpoint_name = 'epoch_{epoch}_loss-{training_loss_epoch:.1e}'
    logger_dir = f'logs/{experiment}'

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_weights_only=False, every_n_epochs=100, filename=checkpoint_name)
    logger = TensorBoardLogger(logger_dir, name="default", flush_secs=5)

    model = Model(maestro_config)

    trainer = Trainer(log_every_n_steps=1, max_epochs=args.epochs, devices=1, accelerator="gpu",
                      default_root_dir="/content/runs", logger=logger,
                      callbacks=[checkpoint_callback], enable_progress_bar=True)

    trainer.fit(model, train_generator, val_dataloaders=(valid_generator if args.validate else None))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--validate', dest='validate', action='store_true')
    parser.add_argument('--no-validate', dest='validate', action='store_false')
    parser.set_defaults(validate=True)
    args = parser.parse_args()
    main(args)

