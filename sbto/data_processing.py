import hydra

from sbto.data.aggregate import aggregate_top_samples
from sbto.main import instantiate_from_cfg

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    sim, task, _ = instantiate_from_cfg(cfg)
    
    aggregate_top_samples(
        sim,
        task,
        data_dir=cfg.data.dataset_dir,
        N_top_samples=cfg.data.N_top,
        min_iteration=cfg.data.min_it,
    )
    
if __name__ == "__main__":
    main()