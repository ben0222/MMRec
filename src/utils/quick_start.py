# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def _json_safe(value):
    """Convert configuration and metric values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_better(candidate, incumbent, bigger):
    if incumbent is None:
        return True
    return candidate > incumbent if bigger else candidate < incumbent


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)
    resolved_config = _json_safe(dict(config.final_config_dict))

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    # Retained for backwards-compatible console output.  It intentionally
    # records the historical protocol, which chooses a hyperparameter setting
    # using a test metric.  Structured output below separately selects by
    # validation metric for experiment records.
    best_test_value = 0.0
    idx = best_test_idx = 0
    best_valid_value = None
    best_valid_idx = None
    combination_results = []

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations: all possible combinations of hyper parameters
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model, mg)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))
        combination_result = {
            'index': idx,
            'hyperparameters': dict(zip(config['hyper_parameters'], _json_safe(hyper_tuple))),
            'resolved_config': _json_safe(dict(config.final_config_dict)),
            'best_valid_score': _json_safe(best_valid_score),
            'best_valid_epoch': trainer.best_valid_epoch,
            'valid': _json_safe(best_valid_result),
            'test_upon_valid': _json_safe(best_test_upon_valid),
        }
        combination_results.append(combination_result)
        if _is_better(best_valid_score, best_valid_value, config['valid_metric_bigger']):
            best_valid_value = best_valid_score
            best_valid_idx = idx

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

    return {
        'schema_version': 1,
        'evaluation_protocol': {
            'trainer_epoch_selection': 'validation_metric',
            'legacy_console_hyperparameter_selection': 'test_metric',
            'recorded_hyperparameter_selection': 'validation_metric',
            'note': 'Training and evaluation behaviour is unchanged in phase one.',
        },
        'model': config['model'],
        'dataset': config['dataset'],
        'valid_metric': val_metric,
        'valid_metric_bigger': config['valid_metric_bigger'],
        'resolved_config': resolved_config,
        'hyperparameter_names': list(config['hyper_parameters']),
        'combinations': combination_results,
        'best_by_validation_index': best_valid_idx,
        'legacy_best_by_test_index': best_test_idx,
    }
