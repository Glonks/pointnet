import wandb


class Logger:
    def __init__(self, project, config):
        wandb.init(project=project, config=vars(config))  # type: ignore

        self.last_overwrite = False

    def log(
        self,
        metrics: dict,
        overwrite = False
    ):
        if self.last_overwrite and not overwrite:
            print()
        self.last_overwrite = overwrite

        parts = []
        wandb_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                spec = '.4f'
            else:
                spec = ''

            parts.append(f'{key}: {value:{spec}}')
            wandb_metrics[key] = value

        log_string = ' | '.join(parts)

        if overwrite:
            print('\r' + log_string, end='')
        else:
            print(log_string)
            wandb.log(wandb_metrics)
