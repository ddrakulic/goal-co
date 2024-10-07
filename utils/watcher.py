"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import logging
import os
import warnings
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import yaml


class MetricsLogger:

    '''
    A watcher to record metrics or histograms easily and send them to the backend of choice
    (tensorboard, weights and biases, mlflow) easily and with the same code.

    w = MetricsWatcher(default_send_freq=20, default_send_freq_hist=100, <config>)
    w.record({'loss': loss, 'acc': acc}, timestep=step, histogram_values={'fc0_grad': fc0_grad_values})

    # it is also possible to add metric(s)/histogram(s) with send_freq different than the default (for example for eval after whole loop)
    w.add_metrics(['loss_eval', 'acc_eval'], send_freq=1)
    w.record({'loss_eval': loss_eval, 'acc_eval': acc_eval}, timestep=step)

    # or -1 for 'never send', then send manually
    w.add_metrics(['loss_eval', 'acc_eval'], send_freq=-1)
    for mb in iterator_eval():
        ... calculate loss_eval
        w.record({'loss_eval': loss_eval, 'acc_eval': acc_eval}; timestep=step)
    w.send_now(['loss_eval', 'acc_eval'])

    Note: histograms work only with tensorboard and weights&biases.
    '''

    def __init__(self, default_send_freq=1, default_send_freq_hist=1, backends='1000',
                 project_name: str = None, group_name: str = None, exp_name: Optional[str] = None, exp_res_dir: Optional[str] = None,
                 hps: Optional[Dict] = None, mlflow_tracking_uri: Optional[str] = None,
                 wandb_require_service: Optional[bool] = False, aimstack_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        backends: 'stdout, tb, wandb, mlflow', for example '0100' for no_stdout, tensorboard, no wandb, no mlflow
                   format also accepted: '0-1-1-0'
        required for tensorboard: exp_res_dir
        required for wandb: exp_group_name, exp_name, exp_res_dir, optional: hps
        required for mlflow: mlflow_tracking_uri or exp_res_dir (where to log), exp_group_name, exp_name,
                             optionally hps

        Exps are arranged in a hierarchical structure project/group/exp
        * tb, mlflow, and aim support only 2 hierarchical level: project_name/{group_name}_{exp_name}
        * wandb supports also group: project_name/group_name/exp_name
        """
        self.metrics = {}
        self.histograms = {}
        self.default_send_freq = default_send_freq
        self.default_send_freq_hist = default_send_freq_hist
        self.project_name = project_name
        self.group_name = group_name
        self.exp_name = exp_name
        assert os.path.isdir(exp_res_dir)
        self.exp_res_dir = exp_res_dir
        self.aimstack_dir = aimstack_dir
        self.hps = hps
        if self.hps is not None:
            with open(os.path.join(exp_res_dir, 'hps.yaml'), 'w') as f:
                yaml.dump(self.hps, f)
        self.logger = logger
        self.mlflow_tracking_uri = mlflow_tracking_uri if mlflow_tracking_uri is not None \
            else os.path.join(exp_res_dir, 'mlruns')
        self.wandb_require_service = wandb_require_service
        self.config_funcs = [
            self._config_stdout, self._config_tensorboard, self._config_wandb,
            self._config_mlflow, self._config_aimstack]
        self.send_funcs = [
            self._send_stdout, self._send_tensorboard, self._send_wandb, self._send_mlflow, self._send_aimstack]
        self._config_backends(backends)

    def _config_backends(self, backends):
        # backends: str or int, length 4 or 5 if int, str format also accepted: '1-0-0-1' or '1-0-0-1-1'
        # (we still support length 4 for backward compatibility before aimstack was added)
        if type(backends) == str:
            backends = backends.replace('-', '')
        backends = str(backends)
        assert len(backends) in [4, 5]
        assert sum([c in ['0', '1'] for c in backends]) in [4, 5]
        self.using_backends = [bool(int(s)) for s in backends]
        for using_backend, config_func in zip(self.using_backends, self.config_funcs):
            if using_backend:
                config_func()

    def __getitem__(self, key):
        return self.metrics[key]

    def add_metric(self, name, send_freq=None, exist_ok=False):
        return self._add_metric_or_histogram(name, send_freq, exist_ok)

    def add_histogram(self, name, send_freq=None, exist_ok=False):
        return self._add_metric_or_histogram(name, send_freq, exist_ok, is_histogram=True)

    def add_metrics(self, names: List, send_freq=None, exist_ok=False):
        # normally this is not used as metrics as 'added' automatically the first time they
        # are recorded with record(). Adding a metric manually can be useful only if you want a different log_freq.
        for mname in names:
            self.add_metric(mname, send_freq, exist_ok)

    def add_histograms(self, names: List, send_freq=None, exist_ok=False):
        # normally this is not used as histograms are 'added' automatically the first time they are
        # recorded with record(). Adding a histogram manually can be useful only if you want a different log_freq.
        for hname in names:
            self.add_histogram(hname, send_freq, exist_ok)

    def record(self, metrics: Optional[Dict[str, float]], timestep: int,
               histograms_values: Optional[Dict[str, Union[List, np.array]]] = None):
        # todo maybe prefix and suffix to metric keys ?
        # METRICS
        avgd_metrics = {}
        if metrics is not None:
            # add metrics if they do not exist yet
            new_keys = [k for k in metrics.keys() if k not in self.metrics.keys()]
            self.add_metrics(new_keys)
            avgd_metrics = {k: self.metrics[k].record(v) for k, v in metrics.items()}
            # (this has values None for metrics just recorded but not to be sent to backend this step)
            avgd_metrics = {k: v for k, v in avgd_metrics.items() if v is not None}
        # HISTOGRAMS
        aggr_histograms = {}
        if histograms_values is not None:
            # add histograms if they do not exist yet
            new_keys = [k for k in histograms_values.keys() if k not in self.histograms.keys()]
            self.add_histograms(new_keys)
            aggr_histograms = {k: self.histograms[k].record(v) for k, v in histograms_values.items()}
            # (this has values None for histograms just recorded but not to be sent to backend this step)
            aggr_histograms = {k: v for k, v in aggr_histograms.items() if v is not None}
        # SEND BOTH
        if avgd_metrics or aggr_histograms:
            self._send_to_backend(avgd_metrics, timestep, aggr_histograms)

    def record_histograms(self, histograms_values, timestep):
        # just an alias
        self.record(None, timestep, histograms_values)

    def send_now(self, metrics: List[str], timestep: int, histograms: List[str] = None):
        # force send of metric to backend now. metrics: list of metrics names to send.
        # useful for metrics that have send_freq=-1, for example at eval/test.
        # we should only do this for metrics that have send_freq=-1
        avgd_metrics = {}
        if metrics is not None:
            l = [self.metrics[m].send_freq == -1 for m in metrics]
            assert sum(l) == len(l)
            avgd_metrics = {k: self.metrics[k].get_avg_and_reset() for k in metrics}
        aggr_histograms = {}
        if histograms is not None:
            l = [self.histograms[h].send_freq == -1 for h in histograms]
            assert sum(l) == len(l)
            aggr_histograms = {k: self.histograms[k].get_aggr_and_reset() for k in histograms}
        if avgd_metrics or aggr_histograms:
            self._send_to_backend(avgd_metrics, timestep, aggr_histograms)

    def record_and_send_now(self, metrics: Optional[Dict[str, float]], timestep: int,
                            histograms_values: Optional[Dict[str, float]] = None):
        self.record(metrics, timestep, histograms_values)
        self.send_now(list(metrics.keys()), timestep, list(histograms_values.keys()))

    def close(self):
        if self.using_backends[1]:
            self.tb_writer.close()
        if self.using_backends[2]:
            self.wandb_run.finish()
        if self.using_backends[3]:
            import mlflow
            mlflow.end_run()

    def _add_metric_or_histogram(self, name, send_freq, exist_ok, is_histogram=False):

        # metric or histogram
        mh = self.metrics if not is_histogram else self.histograms
        mh_class = Metric if not is_histogram else Histogram
        # if send_freq not specified, use default send_freq
        default_send_freq = self.default_send_freq if not is_histogram else self.default_send_freq_hist

        if name in mh.keys():
            if not exist_ok:
                raise NameError('Metric name already used')
            else:
                return
        else:
            msf = send_freq if send_freq is not None else default_send_freq
            mh[name] = mh_class(name, msf)

    def _send_to_backend(self, averaged_metrics: Optional[Dict[str, float]], timestep: int,
                         aggr_histograms: Optional[Dict[str, np.histogram]] = None):
        assert not (averaged_metrics is None and aggr_histograms is None), \
            'please provide either metric or histogram to log'
        backends_logged = 0
        for using_backend, send_func in zip(self.using_backends, self.send_funcs):
            if using_backend:
                send_func(averaged_metrics, timestep, aggr_histograms)
                backends_logged += 1
        if not backends_logged > 0:
            warnings.warn('did not log to any backend. If using histogram, maybe use tb or wandb?')

    def _config_stdout(self):
        self.st_send_func = self.logger.log if self.logger is not None else print

    def _config_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(log_dir=os.path.join(
            self.exp_res_dir, 'tb', self.project_name, self.exp_name_with_group))
        # Tensorboard hyperparams function does not allow logging hps without logging a metric
        # (to be done at the end of an exp: means that a run that does not go to the end would have no hps logging)
        # let's just log in the beginning, using a placeholder metric. (we don't use this metrics viz function)
        def convert_hps(hps):
            # convert types that are not supported by tensorboard
            from copy import deepcopy
            hps_converted = deepcopy(hps)
            allowed_types = [int, float, str, bool, torch.Tensor]
            for k, v in hps.items():
                if type(v) not in allowed_types:
                    if v is None:
                        v_converted = 'None'
                    elif type(v) is list:
                        v_converted = str(v)
                    else:
                        raise NotImplementedError
                    hps_converted[k] = v_converted
            return hps_converted

        self.tb_writer.add_hparams(convert_hps(self.hps), {'placeholder_metric': 0.}, run_name='hparams')

    def _config_wandb(self):
        # hps: Dict {str: other} for example {'slurm_job_id': 123, 'lr': 0.01, 'comment': 'bla'}
        # the wandb 'id' field is useful for resuming exps (same ids are logged together). In our case where
        # gpu-debug exps are stopped/requeued, we want this. So let's use 'name' attribute as id.
        import wandb
        if self.hps is not None:
            for k, v in self.hps.items():
                if type(v) == str:
                    v = '\'{}\''.format(v)
        if self.wandb_require_service:  # useful for distributed training:
            # https://docs.wandb.ai/guides/track/advanced/distributed-training
            wandb.require("service")
        self.wandb_run = wandb.init(
            name=self.exp_name, project=self.project_name, group=self.group_name, dir=self.exp_res_dir,
            id=self.exp_name, resume=True, config=self.hps, reinit=True)

    def _config_mlflow(self):
        # tracking_uri: HTTP/HTTPS URI for a remote mlflow server, a database connection string,
        # or a local path to log data to a directory.
        import mlflow
        # end last run if there is one
        mlflow.end_run()
        # mlflow 'experiment' = wandb 'project' = a bunch of runs
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.project_name)
        mlflow.start_run(run_name=self.exp_name_with_group)
        if self.hps is not None:
            mlflow.log_params(self.hps)

    def _config_aimstack(self):
        from aim import Run, Repo
        # query Runs for this 'experiment' name to see if an existing Run with same exp_name exists,
        # if so the run should be continued with same hash
        existing_runs_with_matching_name = 0
        run_hash = None
        for run in Repo(self.aimstack_dir).iter_runs():
            exp_name = run.get('exp_name', None)
            if exp_name == self.exp_name:
                existing_runs_with_matching_name += 1
                run_hash = run.hash
        assert existing_runs_with_matching_name <= 1

        self.aimstack_run = Run(
            run_hash=run_hash, repo=self.aimstack_dir, experiment=self.project_name)
        self.aimstack_run['exp_name'] = self.exp_name_with_group
        if self.hps is not None:
            self.aimstack_run['hparams'] = self.hps

    def _send_stdout(self, averaged_metrics: Dict, timestep : int, aggregated_histograms):
        # same signature but does not support histograms
        msg = ''
        msg += 'timestep: {}  ---   '.format(timestep)
        for metric, value in averaged_metrics.items():
            msg += '{}: {}  ---  '.format(metric, value)
        # msg += '\n'
        self.st_send_func(msg)

    def _send_tensorboard(self, averaged_metrics, timestep, aggregated_histograms):
        for tag, val in averaged_metrics.items():
            self.tb_writer.add_scalar(tag, val, global_step=timestep)
        for tag, val in aggregated_histograms.items():
            counts, limits, tb_kwargs = val
            self.tb_writer.add_histogram_raw(tag, **tb_kwargs, global_step=timestep)

    @staticmethod
    def _send_wandb(averaged_metrics, timestep, aggregated_histograms):
        import wandb
        all_metrics = {}
        all_metrics.update(averaged_metrics)
        # remove 'tb_kwargs' from histograms 3rd value field, needed only for tensorboard, and use wandb.Histogram
        aggregated_histograms = {k: wandb.Histogram(np_histogram=(v[0], v[1])) for k, v in aggregated_histograms.items()}
        all_metrics.update(aggregated_histograms)
        all_metrics.update({'timestep': timestep})
        wandb.log(all_metrics)
        # aggregated_histograms_wandb = {k: wandb.Histogram(np_histogram=v) for k, v in aggregated_histograms.items()}
        # wandb.run.summary.update(aggregated_histograms_wandb)

    @staticmethod
    def _send_mlflow(averaged_metrics, timestep, aggregated_histograms):
        # same signature but does not support histograms
        import mlflow
        mlflow.log_metrics(averaged_metrics, timestep)

    def _send_aimstack(self, averaged_metrics, timestep, aggregated_histograms):
        for mk, mv in averaged_metrics.items():
            self.aimstack_run.track(value=mv, name=mk, step=timestep)
        if len(aggregated_histograms) > 0:
            raise RuntimeError('Not implemented yet, todo. ')

    @property
    def exp_name_with_group(self):
        name = self.exp_name
        if self.group_name is not None:
            name = self.group_name + '_' + name
        return name


class Metric:

    # keep a memory of recorded values of a metric in order to send it (to tb, wandb etc..)
    # only once in send_freq steps.

    def __init__(self, name, send_freq):
        self.name = name
        self.send_freq = send_freq
        self.values = []

    def record(self, value):
        avg = None
        self.values.append(value)
        if len(self.values) == self.send_freq:
            avg = self.get_avg_and_reset()
        return avg

    def config_send_freq(self, send_freq):
        self.send_freq = send_freq

    def get_avg_and_reset(self):
        assert len(self.values) >= 1, 'empty values list'
        avg = np.mean(self.values)
        self.values = []
        return avg


class Histogram:

    def __init__(self, name, send_freq):
        self.name = name
        self.send_freq = send_freq
        self.records = []  # each record is a np.array of values

    def record(self, values: Union[List, np.array]):
        res = None
        self.records.append(np.array(values).squeeze())
        if len(self.records) == self.send_freq:
            # aggregate all records
            res = self.get_aggr_hist_and_reset()
        return res

    def config_send_freq(self, send_freq):
        self.send_freq = send_freq

    def get_aggr_hist_and_reset(self):
        # aggregate records and get np histogram, return histogram
        assert len(self.records) >= 1, 'empty records list'
        aggr = np.concatenate(self.records)
        self.records = []
        counts, limits = np.histogram(aggr, density=True)
        tb_kwargs = {'min': aggr.min(), 'max': aggr.max(), 'num': len(aggr),
                     'sum': sum(aggr), 'sum_squares': aggr.dot(aggr),
                     'bucket_limits': limits[1:].tolist(), 'bucket_counts': counts.tolist()}
        # it uses less network to directly pass the histogram to frameworks rather than the raw data.
        # returning aggr is just
        return counts, limits, tb_kwargs


