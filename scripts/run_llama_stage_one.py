import os
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

import sys
import cProfile
sys.path.append('/scratch/xliu112/Mixture-of-Domain-Adapters/')
from src.models.llama_mixda_stage_one import LLaMAMixDAStageOneCLM
import wandb

if __name__ == "__main__":
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:5012'
    parser = ArgumentParser()

    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--project_name", type=str, default="add-fever-complete")
    parser.add_argument("--run_name", type=str, default="test")
    parser = LLaMAMixDAStageOneCLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()

    if args.seed is not None:
        seed_everything(seed=args.seed)

    logger = WandbLogger(project=args.project_name, name=args.run_name)
    
    callbacks = [
        LearningRateMonitor(
            logging_interval="step",
        ),
    ]

    trainer = Trainer.from_argparse_args(
        args,
        precision=16, # newly added
        accumulate_grad_batches=4, # newly added
        logger=logger, 
        callbacks=callbacks,
        strategy='ddp',
        num_sanity_val_steps=0
    )
    
    model = LLaMAMixDAStageOneCLM(**vars(args))
    
    trainer.fit(model)