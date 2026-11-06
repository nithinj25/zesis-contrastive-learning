"""Phase 0 check: read the config with Hydra and log a few fake steps to W&B."""
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    run = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        name="phase0-dummy",
    )
    lr = cfg.optim.lr_base * cfg.optim.batch / 256
    for step in range(5):
        run.log({"dummy/loss": 1.0 / (step + 1), "dummy/lr": lr}, step=step)
    print(f"logged 5 steps; mode={cfg.wandb.mode}, K={cfg.moco.K}, batch={cfg.optim.batch}, lr={lr}")
    run.finish()


if __name__ == "__main__":
    main()
